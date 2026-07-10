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


def test_v2_metric_does_not_import_v1_xt_gk():
    import inspect

    import silly_kicks.xtgk._metric as v2_metric

    src = inspect.getsource(v2_metric)
    assert "tracking._xt_gk" not in src and "tracking/_xt_gk" not in src


def test_v1_output_columns_unchanged():
    # Guard: v2 landing must not change v1 output column names (lakehouse/UI Hyrum).
    from silly_kicks.tracking import _xt_gk as v1

    assert v1._OUTPUT_COLS == ["xt_gk_base", "xt_gk_pev", "xt_gk_rav", "xt_gk_dzv", "xt_gk_pressure", "xt_gk"]


def test_v2_columns_disjoint_from_frozen_v1_columns():
    # H1: v2 must NOT reuse v1's xt_gk_pev/rav/dzv (frozen, lakehouse/UI-read).
    from silly_kicks.tracking import _xt_gk as v1
    from silly_kicks.xtgk import _metric as v2

    assert set(v1._OUTPUT_COLS).isdisjoint(set(v2._OUTPUT_COLS))
