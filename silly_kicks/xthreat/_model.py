"""Expected Threat (xT) model — pluggable transition family. See NOTICE for citations."""

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    _action_prob_from_counts,
    _get_cell_indexes,
    _get_flat_indexes,
    _get_successful_move_actions,
    _scoring_prob,
    _scoring_prob_from_counts,
)
from silly_kicks.xthreat._params import (
    _METHOD_TO_PARAMS_TYPE,
    GridSpec,
    KDEParams,
    Method,
    XtParams,
    validate_params_for_method,
)
from silly_kicks.xthreat._physical import require_fitted_xt
from silly_kicks.xthreat._transitions import _singh_from_counts, singh_transition_matrix
from silly_kicks.xthreat._value_iteration import value_iteration

#: Schema version stamped into ``ExpectedThreat.to_dict`` output; ``from_dict`` fail-closes on any
#: other value (forward-compat door -- a newer schema is rejected loudly by an older reader).
_SERIALIZE_FORMAT_VERSION = 1

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

    #: SPADL action-type names sk counts as a "move" (pass|dribble|cross; take_on excluded) and a
    #: "shot" -- exposed so a counts producer replicates ``fit``'s internal filters exactly
    #: (SK-XT-COUNTS). Sourced from ``spadlconfig.actiontype_id``; the values are the canonical keys.
    MOVE_TYPE_NAMES: tuple[str, str, str] = ("pass", "dribble", "cross")
    SHOT_TYPE_NAME: str = "shot"

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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete fitted state to a JSON-round-trippable dict.

        ndarrays become nested Python lists and NumPy scalars become Python floats, so
        ``json.dumps(xt.to_dict())`` is safe (no pickle). The stored arrays -- ``xT`` in
        particular -- are round-tripped VERBATIM in their internal (y-inverted) storage
        orientation (ADR-041); no normalization is applied on either leg. ``from_dict``
        reconstructs a bit-identical model without re-fitting. See NOTICE for citations.

        Raises
        ------
        NotFittedError
            If the model has not been fitted (all-zero ``xT``).

        Returns
        -------
        dict
            A ``format_version``-tagged, JSON-safe dict of the fitted state.

        Examples
        --------
        Persist a fitted model across a process boundary::

            xt = ExpectedThreat().fit(actions)
            blob = json.dumps(xt.to_dict())
            xt2 = ExpectedThreat.from_dict(json.loads(blob))
        """
        if (
            self.scoring_prob_matrix is None
            or self.shot_prob_matrix is None
            or self.move_prob_matrix is None
            or self.transition_matrix is None
            or not np.any(self.xT)
        ):
            raise NotFittedError("ExpectedThreat.to_dict() on an unfitted model; call fit() first.")
        return {
            "format_version": _SERIALIZE_FORMAT_VERSION,
            "l": int(self.l),
            "w": int(self.w),
            "eps": float(self.eps),
            "method": self.method,
            "params": (asdict(self.params) if self.params is not None else None),
            "xT": self.xT.tolist(),
            "scoring_prob_matrix": self.scoring_prob_matrix.tolist(),
            "shot_prob_matrix": self.shot_prob_matrix.tolist(),
            "move_prob_matrix": self.move_prob_matrix.tolist(),
            "transition_matrix": self.transition_matrix.tolist(),
            "heatmaps": [h.tolist() for h in self.heatmaps],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExpectedThreat":
        """Reconstruct a fitted ExpectedThreat from :meth:`to_dict` output WITHOUT re-fitting.

        Fail-closed: an unknown/missing ``format_version`` raises ``ValueError`` FIRST; a
        missing required key raises ``KeyError``; a method/params mismatch raises ``TypeError``
        (via ``validate_params_for_method`` in the constructor); an all-zero ``xT`` payload
        fails ``require_fitted_xt`` (``NotFittedError``). Arrays are restored verbatim in their
        stored orientation (ADR-041). See NOTICE for citations.

        Parameters
        ----------
        d : Mapping
            The dict produced by :meth:`to_dict` (or its JSON round-trip).

        Returns
        -------
        ExpectedThreat
            A fitted model producing bit-identical ``rate`` / ``destination_profiles`` output.

        Examples
        --------
        Round-trip a fitted model::

            xt2 = ExpectedThreat.from_dict(xt.to_dict())
            # xt2.rate(actions) equals xt.rate(actions)
        """
        version = d.get("format_version")
        if version != _SERIALIZE_FORMAT_VERSION:
            raise ValueError(
                f"unsupported ExpectedThreat format_version {version!r}; this reader supports "
                f"{_SERIALIZE_FORMAT_VERSION}."
            )
        raw_params = d.get("params")
        params = None if raw_params is None else _METHOD_TO_PARAMS_TYPE[d["method"]](**raw_params)
        model = cls(l=d["l"], w=d["w"], eps=d["eps"], method=d["method"], params=params)
        model.xT = np.asarray(d["xT"], dtype=np.float64)
        model.scoring_prob_matrix = np.asarray(d["scoring_prob_matrix"], dtype=np.float64)
        model.shot_prob_matrix = np.asarray(d["shot_prob_matrix"], dtype=np.float64)
        model.move_prob_matrix = np.asarray(d["move_prob_matrix"], dtype=np.float64)
        model.transition_matrix = np.asarray(d["transition_matrix"], dtype=np.float64)
        model.heatmaps = [np.asarray(h, dtype=np.float64) for h in d["heatmaps"]]
        require_fitted_xt(model, caller="from_dict")
        return model

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write :meth:`to_dict` as a UTF-8 JSON file (thin wrapper over ``to_dict``).

        Examples
        --------
        Persist a fitted model to disk::

            xt = ExpectedThreat().fit(actions)
            xt.save("xt.json")
        """
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ExpectedThreat":
        """Read a JSON file written by :meth:`save` and reconstruct via :meth:`from_dict`.

        Examples
        --------
        Reload a persisted model and score with it::

            xt = ExpectedThreat.load("xt.json")
            values = xt.rate(actions)
        """
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def fit_from_counts(
        self,
        *,
        shot_counts: npt.NDArray[np.integer],
        goal_counts: npt.NDArray[np.integer],
        move_counts: npt.NDArray[np.integer],
        transition_start_counts: npt.NDArray[np.integer],
        transition_counts: npt.NDArray[np.integer],
        params: XtParams | None = None,
    ) -> "ExpectedThreat":
        """Fit from pre-aggregated zone counts instead of a raw action DataFrame (SK-XT-COUNTS).

        Builds the 4 probability matrices from the supplied integer counts with the SAME raw
        ``_safe_divide`` / row-normalisation ``fit(actions)`` applies (there is NO smoothing/prior),
        then runs the identical ``value_iteration`` -> ``self.xT`` / ``self.heatmaps``. Numerically
        identical to ``ExpectedThreat(l, w).fit(actions)`` when the counts are the exact aggregates of
        ``actions``. Because all inputs are sums, they are ADDITIVE across partitions/competitions
        (``global`` counts = element-wise sum of per-competition counts) -- the property that lets a
        producer fit in one distributed pass without pulling rows to a driver.

        Counts must be computed on LTR-oriented actions (ADR-041), binned with :meth:`zones_of` /
        :meth:`flat_indexes_of`. **Singh (count-based) transition only** -- a KDE request raises.

        Parameters
        ----------
        shot_counts, goal_counts, move_counts : NDArray, shape (w, l)
            Shots / goals(from shots) / ALL moves (``MOVE_TYPE_NAMES``, any result) originating per
            zone, valid START. ``move_counts`` feeds the shoot-vs-move choice.
        transition_start_counts : NDArray, shape (w, l)
            Moves with a valid START *and* END (success + fail) per start zone -- the Singh row
            denominator (DIFFERENT population from ``move_counts``).
        transition_counts : NDArray, shape (w*l, w*l)
            SUCCESSFUL moves ``flat(from) -> flat(to)`` (:meth:`flat_indexes_of`) -- the Singh numerator.
        params : XtParams or None
            Singh/default only; a KDE method/params raises ``ValueError``.

        Raises
        ------
        ValueError
            If ``params`` requests KDE, or any count array has the wrong shape.

        Examples
        --------
        Fit the global grid from summed per-competition counts (one distributed pass)::

            counts = spark_aggregate_zone_counts(all_actions)   # 5 count arrays, additive
            xt = ExpectedThreat(l=16, w=12).fit_from_counts(**counts)
            # xt.xT equals ExpectedThreat(16, 12).fit(all_actions).xT (within fp tolerance)
        """
        if self.method == "kde_smoothed" or isinstance(params, KDEParams):
            raise ValueError(
                "KDE transition is not a pure count aggregate; use fit(actions) for KDE, or "
                "fit_from_counts with singh/default params."
            )
        n = self.w * self.l
        for name, arr, shape in (
            ("shot_counts", shot_counts, (self.w, self.l)),
            ("goal_counts", goal_counts, (self.w, self.l)),
            ("move_counts", move_counts, (self.w, self.l)),
            ("transition_start_counts", transition_start_counts, (self.w, self.l)),
            ("transition_counts", transition_counts, (n, n)),
        ):
            got = np.asarray(arr).shape
            if got != shape:
                raise ValueError(f"{name} has shape {got}, expected {shape}")
        sc, gc, mc, tsc, tc = (
            np.asarray(x, dtype=np.float64)
            for x in (shot_counts, goal_counts, move_counts, transition_start_counts, transition_counts)
        )
        self.scoring_prob_matrix = _scoring_prob_from_counts(gc, sc)
        self.shot_prob_matrix, self.move_prob_matrix = _action_prob_from_counts(sc, mc)
        self.transition_matrix = _singh_from_counts(tc, tsc.ravel())
        self.xT, self.heatmaps = value_iteration(
            self.scoring_prob_matrix,
            self.shot_prob_matrix,
            self.move_prob_matrix,
            self.transition_matrix,
            eps=self.eps,
        )
        return self

    def zones_of(self, xs: npt.ArrayLike, ys: npt.ArrayLike) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Bin SPADL ``(x, y)`` coordinates to ``(zone_x, zone_y)`` -- sk's exact internal binning.

        ``zone_x = clip(int(x / field_length * l), 0, l-1)``, ``zone_y`` analogously. Exposed so a
        counts producer replicates the binning `fit` uses (SK-XT-COUNTS); see :meth:`flat_indexes_of`
        for the y-inverted flat index the transition counts use.

        Examples
        --------
        Bin an array of coordinates::

            xt = ExpectedThreat(l=16, w=12)
            zx, zy = xt.zones_of([0.0, 105.0], [0.0, 68.0])
        """
        xi, yj = _get_cell_indexes(pd.Series(xs), pd.Series(ys), self.l, self.w)
        return xi.to_numpy(), yj.to_numpy()

    def flat_indexes_of(self, xs: npt.ArrayLike, ys: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """Flat zone index of SPADL ``(x, y)`` -- ``(w-1 - zone_y)*l + zone_x`` (y-INVERTED, ADR-041).

        This is the row/column ordering of ``transition_matrix`` / ``transition_counts``: row 0 is the
        top of the pitch. A counts producer MUST index its ``transition_counts`` with this exact
        formula, or the grid transposes in y silently.

        Examples
        --------
        Flat index for the transition matrix::

            xt = ExpectedThreat(l=16, w=12)
            flat = xt.flat_indexes_of([10.0, 90.0], [20.0, 50.0])
        """
        return _get_flat_indexes(pd.Series(xs), pd.Series(ys), self.l, self.w).to_numpy()
