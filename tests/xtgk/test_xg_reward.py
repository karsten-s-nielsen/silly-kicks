import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk._xg_reward import xg_scoring_prob
from silly_kicks.xthreat._grid import _get_flat_indexes
from tests.xtgk.conftest import FAIL, PASS, SHOT, SUCCESS, _row, make_cohort


def _flat(x, y):
    return int(_get_flat_indexes(pd.Series([float(x)]), pd.Series([float(y)]), 16, 12).iloc[0])


def test_mean_xg_over_shots_per_cell_not_goal_gated():
    # two shots same cell (100,34), xg 0.2 (goal) and 0.4 (miss) -> E[xG|shot]=0.3, NOT 0.2.
    rows = [
        _row(0, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.2),
        _row(1, SHOT, FAIL, 100, 34, 105, 34, xg=0.4),
        _row(2, PASS, SUCCESS, 10, 34, 20, 34, xg=np.nan),
    ]
    surf = xg_scoring_prob(make_cohort(rows), xg_column="xg", l=16, w=12)
    assert surf.shape == (12, 16)
    assert np.isclose(surf.ravel()[_flat(100, 34)], 0.3)


def test_empty_cell_is_zero_not_nan():
    surf = xg_scoring_prob(
        make_cohort([_row(0, PASS, SUCCESS, 10, 34, 20, 34, xg=np.nan)]),
        xg_column="xg",
        l=16,
        w=12,
    )
    assert np.all(np.isfinite(surf)) and np.all(surf == 0.0)


def test_nan_coord_shots_excluded():
    rows = [
        _row(0, SHOT, SUCCESS, np.nan, 34, 105, 34, xg=0.9),
        _row(1, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.3),
    ]
    surf = xg_scoring_prob(make_cohort(rows), xg_column="xg", l=16, w=12)
    assert np.isclose(surf.ravel()[_flat(100, 34)], 0.3)


def test_weighted_sum_layout_matches_count():
    from silly_kicks.xtgk._xg_reward import _weighted_cell_sum
    from silly_kicks.xthreat._grid import _count

    rows = [_row(i, SHOT, SUCCESS, 10 + 7 * i, 20 + 3 * i, 105, 34, xg=1.0) for i in range(8)]
    a = make_cohort(rows)
    assert np.array_equal(
        _count(a.start_x, a.start_y, 16, 12).astype(float),
        _weighted_cell_sum(a.start_x, a.start_y, a["xg"], 16, 12),
    )


def test_missing_xg_column_raises():
    with pytest.raises(ValueError, match="xg_column"):
        xg_scoring_prob(
            make_cohort([_row(0, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.3)]),
            xg_column="nope",
            l=16,
            w=12,
        )
