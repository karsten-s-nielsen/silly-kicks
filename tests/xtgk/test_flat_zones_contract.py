"""T3 -- flat_zones' NaN->0 behaviour is PINNED (the fit seams depend on it), and
finite_coord_mask is the blessed alternative for SCORING callers. ADR-036 amendment."""

import numpy as np
import pandas as pd

from silly_kicks.xtgk._possession_value import finite_coord_mask, flat_zones, zone_of


def test_flat_zones_nan_still_maps_to_zone_176_pinned():
    """PINNED, deliberately. _markov.py:65 / _empirical.py:83 / _diagnostics.py:123 call
    flat_zones WITH NaN rows to assign pressure terciles, then drop them before solving.
    Changing this would silently move every fitted surface."""
    z = flat_zones(pd.Series([float("nan")]), pd.Series([float("nan")]))
    assert int(z[0]) == 176
    assert int(zone_of(0.0, 0.0)) == 176  # NaN -> (0,0) -> the own-corner cell


def test_finite_coord_mask_flags_every_non_finite_coordinate():
    actions = pd.DataFrame(
        {
            "start_x": [5.0, np.nan, 5.0, 5.0, 5.0],
            "start_y": [34.0, 34.0, np.nan, 34.0, 34.0],
            "end_x": [40.0, 40.0, 40.0, np.nan, 40.0],
            "end_y": [34.0, 34.0, 34.0, 34.0, np.inf],
        }
    )
    mask = finite_coord_mask(actions)
    assert mask.tolist() == [True, False, False, False, False]


def test_finite_coord_mask_is_all_true_on_clean_input():
    actions = pd.DataFrame(
        {"start_x": [5.0, 6.0], "start_y": [34.0, 30.0], "end_x": [40.0, 41.0], "end_y": [34.0, 20.0]}
    )
    assert finite_coord_mask(actions).all()
