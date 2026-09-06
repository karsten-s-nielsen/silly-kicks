"""Public destination-distribution seam for counterfactual consumers (e.g. territory TF-54b).

Exposes the fitted xT's transition row + zone geometry + xT values in PHYSICAL coordinates, keeping
the flat-index / y-inversion convention inside xthreat (ADR-041). Returns the RAW transition row;
consumers renormalize over their selected zone subset, so the family-specific row scale cancels
(family-agnostic: singh_counts and kde_smoothed both yield a valid renormalized distribution).
See NOTICE for citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.xthreat._grid import _get_flat_indexes
from silly_kicks.xthreat._physical import require_fitted_xt, values_at_points
from silly_kicks.xthreat._transitions import _zone_centres

__all__ = ["DestinationProfile", "destination_profiles"]


@dataclass(frozen=True)
class DestinationProfile:
    """Per-origin destination geometry + xT values, in PHYSICAL SPADL coordinates (ADR-041).

    Attributes
    ----------
    zone_centres : np.ndarray
        ``(n_zones, 2)`` physical SPADL coordinates of each flat-index zone centre,
        ascending-y (ADR-041-correct).
    zone_values : np.ndarray
        ``(n_zones,)`` xT value at each zone centre.
    probabilities : np.ndarray
        ``(n_origins, n_zones)`` RAW transition row ``T[origin_cell, :]`` per origin (NOT
        renormalized). Consumers renormalize over their selected zone subset, so the
        family-specific row scale cancels.

    Examples
    --------
    Read one origin's most-likely destination zone::

        from silly_kicks.xthreat import destination_profiles

        prof = destination_profiles(fitted_xt, actions["start_x"], actions["start_y"])
        top_zone = prof.zone_centres[prof.probabilities[0].argmax()]  # (x, y) in metres
    """

    zone_centres: np.ndarray
    zone_values: np.ndarray
    probabilities: np.ndarray


def destination_profiles(model, origin_x, origin_y) -> DestinationProfile:
    """Expose the fitted xT's destination distribution + zone geometry for counterfactual consumers.

    Keeps the flat-index / y-inversion convention inside xthreat (ADR-041): callers receive zone
    centres, zone xT values and per-origin transition rows in PHYSICAL SPADL coordinates and never
    touch the raw flat-indexed, y-inverted ``.transition_matrix`` / ``.xT``.

    Parameters
    ----------
    model : ExpectedThreat
        A fitted xT model. Fails closed via ``require_fitted_xt`` on ``None``, a variant-name
        ``str`` or an unfitted model.
    origin_x : np.ndarray or pd.Series
        Action-LTR x coordinates (metres) of each origin.
    origin_y : np.ndarray or pd.Series
        Action-LTR y coordinates (metres) of each origin.

    Returns
    -------
    DestinationProfile
        ``zone_centres`` ``(n_zones, 2)``, ``zone_values`` ``(n_zones,)`` and ``probabilities``
        ``(n_origins, n_zones)`` — the RAW transition row per origin.

    Examples
    --------
    Renormalize an origin's destination distribution over a selected zone subset::

        from silly_kicks.xthreat import destination_profiles

        prof = destination_profiles(fitted_xt, actions["start_x"], actions["start_y"])
        row = prof.probabilities[0]
        subset = row > 0
        q = row[subset] / row[subset].sum()  # a valid distribution, family-agnostic
    """
    require_fitted_xt(model, caller="destination_profiles")
    centres = _zone_centres(model.grid)  # (n_zones, 2) physical, ADR-041-correct
    values = values_at_points(model, centres[:, 0], centres[:, 1])
    ox = pd.Series(np.asarray(origin_x, dtype=float))
    oy = pd.Series(np.asarray(origin_y, dtype=float))
    cells = _get_flat_indexes(ox, oy, model.l, model.w).to_numpy()
    probabilities = np.asarray(model.transition_matrix, dtype=float)[cells]
    return DestinationProfile(zone_centres=centres, zone_values=values, probabilities=probabilities)
