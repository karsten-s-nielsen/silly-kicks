import numpy as np

from silly_kicks.xtgk._validate import validate_possession_value_input
from tests.xtgk.conftest import (
    GOALKICK,
    SHOT,
    SUCCESS,
    _row,
    make_cohort,
    mirror_x,
    three_band_cohort,
)


def _ok():
    return make_cohort(
        [
            _row(0, GOALKICK, SUCCESS, 5, 34, 60, 34, xg=np.nan, pressure=0.2),
            _row(1, SHOT, SUCCESS, 100, 34, 105, 34, xg=0.3, pressure=0.8),
        ]
    )


def test_ok_input_passes():
    diag = validate_possession_value_input(_ok(), xg_column="xg", pressure_column="pressure")
    assert diag.ok is True and diag.problems == []


def test_missing_columns_flagged():
    diag = validate_possession_value_input(_ok(), xg_column="missing", pressure_column="pressure")
    assert diag.ok is False and any("missing" in p for p in diag.problems)


def test_attack_reversed_orientation_flagged():
    diag = validate_possession_value_input(mirror_x(three_band_cohort(20)), xg_column="xg", pressure_column="pressure")
    assert diag.ok is False and any("orientation" in p.lower() for p in diag.problems)


def test_require_possession_id_for_crosscheck():
    diag = validate_possession_value_input(
        _ok().drop(columns=["possession_id"]),
        xg_column="xg",
        pressure_column="pressure",
        require_possession_id=True,
    )
    assert diag.ok is False and any("possession_id" in p for p in diag.problems)
