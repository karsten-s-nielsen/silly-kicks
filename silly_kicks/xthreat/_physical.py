"""Physical-coordinate adapters over a fitted xT grid (ADR-041).

``ExpectedThreat.xT`` stores rows y-INVERTED (row 0 = the TOP of the pitch; see
``_grid.py``), and ``ExpectedThreat.interpolator()`` preserves that storage orientation in
its output -- ``rate()`` compensates with the same inverted indexing, but a consumer that
multiplies the raw interpolator output against an ascending-y grid gets a silent y-mirror.

This module is the ONE place that inversion is neutralized, and the single home of the
fitted-model guard. It lives in ``xthreat`` because the inversion is xthreat's own storage
convention: the anti-corruption layer belongs on this boundary, not on each consumer's.

Must NOT import ``silly_kicks.vaep`` (cycle: vaep consumes this module).
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import _get_cell_indexes

if TYPE_CHECKING:  # duck-typed at runtime (reads only model.xT)
    from silly_kicks.xthreat._model import ExpectedThreat

__all__ = ["physical_grid", "require_fitted_xt", "values_at_points"]


def require_fitted_xt(model: ExpectedThreat | str | None, *, caller: str) -> None:
    """Fail closed unless ``model`` is a fitted ExpectedThreat.

    The single source for this guard: ``vaep.features.xt_xfns``, its atomic mirror and the
    tracking adapters all call it (ADR-041 -- two duplicated copies collapsed here).

    Parameters
    ----------
    model : ExpectedThreat or str or None
        The candidate xT model.
    caller : str
        Public name used in the messages (e.g. ``"xt_xfns"``, ``"add_obso"``).

    Raises
    ------
    NotImplementedError
        If ``model`` is a ``str`` (a future bundled-variant name; not shipped yet).
    ValueError
        If ``model`` is ``None``.
    NotFittedError
        If ``model`` is an unfitted ExpectedThreat (all-zero ``.xT``).

    Examples
    --------
    Guard a caller-supplied model::

        from silly_kicks.xthreat import require_fitted_xt

        require_fitted_xt(model, caller="my_feature")
    """
    if isinstance(model, str):
        raise NotImplementedError(
            f"{caller}: bundled xT grid variants are not shipped yet; pass a fitted ExpectedThreat."
        )
    if model is None:
        raise ValueError(f"{caller} requires a fitted ExpectedThreat (model=...).")
    if not np.any(model.xT):  # same fitted-check ExpectedThreat.rate() uses
        raise NotFittedError(f"{caller} requires a fitted ExpectedThreat; call model.fit(actions) first.")


def physical_grid(
    model: ExpectedThreat | str | None,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    *,
    require_fitted: bool = True,
) -> np.ndarray:
    """Sample a fitted xT surface onto ascending-y, ascending-x cell centres (metres).

    Returns a ``(len(grid_y), len(grid_x))`` grid with row 0 = the LOWEST physical y --
    the pitch-control / OBSO convention -- and +x = the attacking direction (xT is fit on
    action-LTR SPADL actions). Orientation is normalized at the DATA level (``np.flipud``,
    so row 0 becomes y=0) BEFORE spline construction, which is why any ascending
    ``grid_y`` is valid and no grid-symmetry precondition is needed.

    .. important::
        **Pass the CONSUMER's own grid; do not invent one.** Every correct caller hands in
        the grid its consumer already uses (``pc.grid_x`` / ``pc.grid_y``), which makes the
        registration correct by construction. The one site that invented a grid drifted
        immediately -- it built cell centres while its consumer was node-registered, a
        systematic +-0.5 m offset (ADR-041). If you must construct one, match the
        consumer's convention exactly and pin it with a contract test.

    Parameters
    ----------
    model : ExpectedThreat
        A fitted xT model.
    grid_x : np.ndarray
        1-D strictly ascending x sample coordinates in metres, in the CONSUMER's
        registration (node or cell-centre -- this function samples wherever it is told).
    grid_y : np.ndarray
        1-D strictly ascending y sample coordinates in metres, same convention.
    require_fitted : bool, default True
        When ``False``, relax ONLY the all-zero-grid check, for callers whose own
        documented contract is to degrade rather than raise on a degenerate surface:
        ``compute_gk_influence`` and ``compute_blocking_score`` both return NaN there
        (pinned by ``test_gk_influence.py::TestXtOrientation::test_xt_all_zeros_returns_nan``),
        and the calibration harness legitimately fits an all-zero grid from a slim corpus.
        ``None`` and a variant-name ``str`` still fail closed — those are misuse under
        every contract. This exists so the ADR-041 orientation repair can be SHARED without
        silently importing a fail-closed policy into modules that never had one.

    Returns
    -------
    np.ndarray
        ``(len(grid_y), len(grid_x))`` xT values.

    Raises
    ------
    ValueError
        If ``model`` is ``None``, or the grids are not 1-D strictly ascending with at
        least two cell centres each.

    Examples
    --------
    Build an EPV grid matching a pitch-control surface::

        from silly_kicks.xthreat import physical_grid

        epv = physical_grid(fitted_xt, pc.grid_x, pc.grid_y)  # (ny, nx)
    """
    if require_fitted or model is None or isinstance(model, str):
        require_fitted_xt(model, caller="physical_grid")

    gx = np.asarray(grid_x, dtype=float)
    gy = np.asarray(grid_y, dtype=float)
    if gx.ndim != 1 or gy.ndim != 1 or gx.size < 2 or gy.size < 2:
        raise ValueError("physical_grid: grid_x and grid_y must be 1-D with at least 2 cell centres")
    if np.any(np.diff(gx) <= 0) or np.any(np.diff(gy) <= 0):
        raise ValueError("physical_grid: grid_x and grid_y must be strictly ascending (metres)")

    xT = np.asarray(model.xT, dtype=float)  # type: ignore[union-attr]
    w, l = xT.shape
    cell_length = spadlconfig.field_length / l
    cell_width = spadlconfig.field_width / w
    x_centres = np.arange(0.0, spadlconfig.field_length, cell_length) + 0.5 * cell_length
    y_centres = np.arange(0.0, spadlconfig.field_width, cell_width) + 0.5 * cell_width

    # THE inversion-neutralization point: storage row 0 is the TOP of the pitch, so flip
    # the DATA once here rather than compensating in every consumer.
    phys = np.flipud(xT)
    spline = RectBivariateSpline(x_centres, y_centres, phys.T, kx=1, ky=1)
    return np.asarray(spline(gx, gy)).T  # (ny, nx), ascending-y rows


def values_at_points(
    model: ExpectedThreat | str | None,
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    *,
    require_fitted: bool = True,
) -> np.ndarray:
    """Per-point xT at action-LTR coords -- exact ``rate(use_interpolation=False)`` semantics.

    Nearest-cell lookup through the frozen cell indexer with the ``(n_rows-1)-yj`` row
    inversion. NaN coordinates map to NaN rather than raising (real provider data carries
    NaN coords; ``_get_cell_indexes`` casts to int64 and would raise).

    Parameters
    ----------
    model : ExpectedThreat
        A fitted xT model.
    x : np.ndarray or pd.Series
        Action-LTR x coordinates in metres.
    y : np.ndarray or pd.Series
        Action-LTR y coordinates in metres.
    require_fitted : bool, default True
        When ``False``, relax ONLY the all-zero-grid check — see :func:`physical_grid` for
        the full rationale. ``None`` and a variant-name ``str`` still fail closed.

    Returns
    -------
    np.ndarray
        xT value per point; NaN where either coordinate is NaN.

    Examples
    --------
    Value a completed pass::

        from silly_kicks.xthreat import values_at_points

        gain = values_at_points(xt, a["end_x"], a["end_y"]) - values_at_points(
            xt, a["start_x"], a["start_y"]
        )
    """
    # Same opt-out contract as physical_grid: relax ONLY the all-zero-grid check, for callers
    # whose own contract is to degrade rather than raise (compute_blocking_score; the
    # calibration harness legitimately fits an all-zero grid from a slim corpus). None and a
    # variant-name str still fail closed.
    if require_fitted or model is None or isinstance(model, str):
        require_fitted_xt(model, caller="values_at_points")

    grid = np.asarray(model.xT, dtype=float)  # type: ignore[union-attr]
    n_rows, n_cols = grid.shape
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    result = np.full(xa.shape, np.nan, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    if valid.any():
        xi, yj = _get_cell_indexes(pd.Series(xa[valid]), pd.Series(ya[valid]), n_cols, n_rows)
        result[valid] = grid[(n_rows - 1) - yj.to_numpy(), xi.to_numpy()]
    return result
