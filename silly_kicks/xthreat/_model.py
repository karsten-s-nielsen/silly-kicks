"""Expected Threat (xT) model — pluggable transition family. See NOTICE for citations."""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import RectBivariateSpline  # type: ignore[reportMissingImports]
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import (
    M,
    N,
    _action_prob,
    _get_cell_indexes,
    _get_successful_move_actions,
    _scoring_prob,
)
from silly_kicks.xthreat._params import GridSpec, KDEParams, Method, XtParams, validate_params_for_method
from silly_kicks.xthreat._transitions import singh_transition_matrix
from silly_kicks.xthreat._value_iteration import value_iteration

# NOTE: kde_smoothed_transition_matrix is lazy-imported inside fit() (below), NOT at module
# top. This (a) lets the package import cleanly before KDE lands and (b) keeps `import
# silly_kicks` light (sklearn is only pulled when the KDE path actually runs).


class ExpectedThreat:
    """An implementation of the Expected Threat (xT) model [1]_.

    The xT model values actions that successfully move the ball between two locations by the
    difference in long-term scoring probability between the start and end location.

    ``method="kde_smoothed"`` swaps the transition builder only; ``method="singh_counts"``
    (the default) is byte-identical to the classic implementation.

    Parameters
    ----------
    l : int
        Grid cells in the x-dimension (default 16). Maps to ``GridSpec(n_zones_x=l, ...)``.
    w : int
        Grid cells in the y-dimension (default 12). Maps to ``GridSpec(n_zones_y=w)``.
    eps : float
        Value-iteration precision (default 1e-5).
    method : {"singh_counts", "kde_smoothed"}
        Transition family. Default "singh_counts".
    params : SinghParams | KDEParams | None
        Method parameters; validated against ``method``. ``None`` uses that method's defaults.

    References
    ----------
    .. [1] Singh, Karun. "Introducing Expected Threat (xT)." 15 February, 2019.
        https://karun.in/blog/expected-threat.html

    Examples
    --------
    Fit an Expected Threat (xT) grid and rate actions::

        from silly_kicks.xthreat import ExpectedThreat

        xt = ExpectedThreat()
        xt.fit(actions)
        values = xt.rate(actions)  # ndarray of shape (len(actions),)

    KDE-smoothed at a higher resolution::

        from silly_kicks.xthreat import ExpectedThreat, KDEParams

        xt = ExpectedThreat(l=24, w=16, method="kde_smoothed", params=KDEParams()).fit(actions)
    """

    def __init__(
        self,
        l: int = N,
        w: int = M,
        eps: float = 1e-5,
        method: Method = "singh_counts",
        params: XtParams | None = None,
    ) -> None:
        validate_params_for_method(method, params)
        self.l = l
        self.w = w
        self.eps = eps
        self.method: Method = method
        self.params = params
        self.grid = GridSpec(n_zones_x=l, n_zones_y=w)
        self.heatmaps: list[npt.NDArray[np.float64]] = []
        self.xT: npt.NDArray[np.float64] = np.zeros((self.w, self.l))
        self.scoring_prob_matrix: npt.NDArray[np.float64] | None = None
        self.shot_prob_matrix: npt.NDArray[np.float64] | None = None
        self.move_prob_matrix: npt.NDArray[np.float64] | None = None
        self.transition_matrix: npt.NDArray[np.float64] | None = None

    def fit(self, actions: pd.DataFrame) -> "ExpectedThreat":
        """Fit the xT model with the given actions. See NOTICE for full bibliographic citations.

        Parameters
        ----------
        actions : pd.DataFrame
            Actions, in SPADL format.

        Returns
        -------
        self
            Fitted xT model.

        Examples
        --------
        Fit the xT grid on a SPADL action stream::

            xt = ExpectedThreat().fit(actions)
            # xt.xT is the (W, L) value surface; xt.heatmaps records each iteration.
        """
        self.scoring_prob_matrix = _scoring_prob(actions, self.l, self.w)
        self.shot_prob_matrix, self.move_prob_matrix = _action_prob(actions, self.l, self.w)
        if self.method == "singh_counts":
            self.transition_matrix = singh_transition_matrix(actions, self.grid)
        else:  # kde_smoothed
            from silly_kicks.xthreat._transitions import kde_smoothed_transition_matrix

            params = self.params if isinstance(self.params, KDEParams) else KDEParams()
            self.transition_matrix = kde_smoothed_transition_matrix(actions, self.grid, params)
        self.xT, self.heatmaps = value_iteration(
            self.scoring_prob_matrix,
            self.shot_prob_matrix,
            self.move_prob_matrix,
            self.transition_matrix,
            eps=self.eps,
        )
        return self

    def interpolator(
        self, kind: str = "linear"
    ) -> Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """Interpolate over the pitch. See NOTICE for full bibliographic citations.

        Wraps :class:`scipy.interpolate.RectBivariateSpline` (the SciPy-recommended
        bug-for-bug compatible replacement for the legacy ``interp2d``, removed in
        SciPy 1.14.0). Preserves the legacy ``interp(xs, ys)`` calling convention
        that returns a ``(len(ys), len(xs))`` array — y on the first axis, x on
        the second — matching how callers index the result via
        ``grid[y_indices, x_indices]``.

        Parameters
        ----------
        kind : {'linear', 'cubic', 'quintic'}  # noqa: DAR103
            The kind of spline interpolation to use. Default is 'linear'.
            Maps to ``RectBivariateSpline(kx=ky=k)`` with k=1/3/5 respectively.

        Raises
        ------
        ImportError
            If scipy is not installed.

        Returns
        -------
        callable
            A function ``interp(xs, ys) -> grid`` that interpolates xT values
            over the pitch. ``xs`` has shape ``(L,)``, ``ys`` has shape ``(W,)``,
            and the returned grid has shape ``(W, L)`` — y-major, matching the
            xT grid's row-major orientation.

        Examples
        --------
        Interpolate xT values across continuous coordinates::

            interp = xt.interpolator(kind="linear")
            grid = interp(xs, ys)  # (len(ys), len(xs)) array — y on first axis.
        """
        if RectBivariateSpline is None:
            raise ImportError("Interpolation requires scipy to be installed.")

        cell_length = spadlconfig.field_length / self.l
        cell_width = spadlconfig.field_width / self.w

        x = np.arange(0.0, spadlconfig.field_length, cell_length) + 0.5 * cell_length
        y = np.arange(0.0, spadlconfig.field_width, cell_width) + 0.5 * cell_width

        # self.xT has shape (w, l) = (y, x). RectBivariateSpline expects z with
        # shape (len(x), len(y)), so transpose the input grid to (l, w) = (x, y).
        k = {"linear": 1, "cubic": 3, "quintic": 5}[kind]
        spline = RectBivariateSpline(x, y, self.xT.T, kx=k, ky=k)

        def _interp(xs: npt.NDArray[np.float64], ys: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            # spline(xs, ys) returns shape (len(xs), len(ys)) — RectBivariateSpline
            # convention. Transpose to (len(ys), len(xs)) to match the legacy
            # interp2d output shape that downstream callers depend on.
            return np.asarray(spline(xs, ys)).T

        return _interp

    def rate(self, actions: pd.DataFrame, use_interpolation: bool = False) -> npt.NDArray[np.float64]:
        """Compute the xT values for the given actions. See NOTICE for citations.

        xT should only be used to value actions that move the ball and also
        keep the current team in possession of the ball. All other actions in
        the given dataframe receive a `NaN` rating.

        Parameters
        ----------
        actions : pd.DataFrame
            Actions, in SPADL format.
        use_interpolation : bool
            Indicates whether to use bilinear interpolation when inferring xT
            values. Note that this requires Scipy to be installed (pip install
            scipy).

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.

        Returns
        -------
        np.ndarray
            The xT value for each action.

        Examples
        --------
        Rate move-class actions in a SPADL stream::

            xt = ExpectedThreat().fit(actions)
            values = xt.rate(actions, use_interpolation=True)
            # Non-move actions (shots / fouls / etc.) receive NaN.
        """
        if not np.any(self.xT):
            raise NotFittedError()

        if not use_interpolation:
            l = self.l
            w = self.w
            grid = self.xT
        else:
            # Use interpolation to create a
            # more fine-grained 1050 x 680 grid
            interp = self.interpolator()
            l = int(spadlconfig.field_length * 10)
            w = int(spadlconfig.field_width * 10)
            xs = np.linspace(0, spadlconfig.field_length, l, dtype=np.float64)
            ys = np.linspace(0, spadlconfig.field_width, w, dtype=np.float64)
            grid = interp(xs, ys)

        ratings = np.empty(len(actions))
        ratings[:] = np.nan

        move_actions = _get_successful_move_actions(actions.reset_index())  # type: ignore[reportArgumentType]
        # Drop actions with NaN coordinates — they cannot be assigned to grid cells.
        move_actions = move_actions.dropna(subset=["start_x", "start_y", "end_x", "end_y"])

        startxc, startyc = _get_cell_indexes(move_actions.start_x, move_actions.start_y, l, w)
        endxc, endyc = _get_cell_indexes(move_actions.end_x, move_actions.end_y, l, w)

        xT_start = grid[startyc.rsub(w - 1), startxc]
        xT_end = grid[endyc.rsub(w - 1), endxc]

        ratings[move_actions.index] = xT_end - xT_start
        return ratings
