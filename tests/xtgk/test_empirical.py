import numpy as np

from silly_kicks.xtgk._empirical import EmpiricalPossessionValue
from silly_kicks.xtgk._possession_value import zone_of
from silly_kicks.xtgk._pressure_levels import PressureLevels
from tests.xtgk.conftest import PASS, SHOT, SUCCESS, _row, make_cohort, three_band_cohort


def test_first_shot_value_matches_terminal_xg_per_tercile():
    a = three_band_cohort()
    pl = PressureLevels().fit(a["pressure"])
    m = EmpiricalPossessionValue().fit(
        a, xg_column="xg", pressure_column="pressure", aggregation="first_shot", pressure_levels=pl
    )
    z = zone_of(3.0, 34.0)
    assert np.isclose(m.value(z, 1), 0.5, atol=1e-9)  # low-pressure band shot xg
    assert np.isclose(m.value(z, 3), 0.05, atol=1e-9)  # high-pressure band shot xg


def test_no_shot_possession_contributes_zero():
    rows = [_row(0, PASS, SUCCESS, 3, 34, 40, 34, possession_id=0, pressure=0.1)]
    m = EmpiricalPossessionValue().fit(
        make_cohort(rows), xg_column="xg", pressure_column="pressure", aggregation="first_shot"
    )
    assert m.value(zone_of(3.0, 34.0), 1) == 0.0


def test_reverse_scan_matches_naive_first_shot():
    from silly_kicks.xtgk._empirical import _possession_outcomes

    rows = [
        _row(0, PASS, SUCCESS, 3, 34, 40, 34, possession_id=0, pressure=0.1),
        _row(1, PASS, SUCCESS, 40, 34, 80, 34, possession_id=0, pressure=0.1),
        _row(2, SHOT, SUCCESS, 90, 34, 105, 34, possession_id=0, pressure=0.1, xg=0.3),
        _row(3, SHOT, SUCCESS, 95, 34, 105, 34, possession_id=0, pressure=0.1, xg=0.7),
    ]
    out = _possession_outcomes(make_cohort(rows), "xg", "first_shot")
    # actions 0,1 -> first shot xg 0.3; action 2 -> next shot 0.7; action 3 -> none 0.0
    assert list(out) == [0.3, 0.3, 0.7, 0.0]
