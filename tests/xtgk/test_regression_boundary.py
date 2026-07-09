"""xtgk touches NO xthreat source; importing it must not change any classic xthreat output."""

import numpy as np

import silly_kicks.xtgk  # noqa: F401
from silly_kicks.xthreat import GridSpec, singh_transition_matrix
from silly_kicks.xthreat._grid import _action_prob
from tests.xtgk.conftest import PASS, SHOT, SUCCESS, _row, make_cohort

GRID = GridSpec(n_zones_x=16, n_zones_y=12)


def test_classic_xt_unaffected_by_xtgk_import():
    rows = [_row(i, PASS, SUCCESS, 10 + i, 34, 50 + i, 40) for i in range(12)]
    rows += [_row(12, SHOT, SUCCESS, 100, 34, 105, 34)]
    a = make_cohort(rows)
    transition = singh_transition_matrix(a, GRID)
    assert transition.shape == (192, 192) and np.isfinite(transition).all()
    s, m = _action_prob(a, 16, 12)
    assert np.isfinite(s).all() and np.isfinite(m).all()
