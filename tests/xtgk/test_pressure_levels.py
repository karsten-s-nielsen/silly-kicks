import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk._pressure_levels import PressureLevels


def test_global_terciles_partition_roughly_thirds():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 900)))
    counts = pd.Series(pl.apply(pd.Series(np.linspace(0, 1, 900)))).value_counts()
    assert set(counts.index) == {1, 2, 3}
    assert all(280 <= counts[k] <= 320 for k in (1, 2, 3))


def test_three_band_input_populates_all_levels():
    pl = PressureLevels(mode="global").fit(pd.Series([0.1, 0.5, 0.9] * 100))
    lv = pl.apply(pd.Series([0.1, 0.5, 0.9]))
    assert set(lv.tolist()) == {1, 2, 3}


def test_apply_stable_to_persisted_cutpoints():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 300)))
    assert pl.cutpoints is not None
    pl2 = PressureLevels.from_cutpoints(pl.cutpoints)
    assert np.array_equal(pl.apply(pd.Series([0.1, 0.5, 0.9])), pl2.apply(pd.Series([0.1, 0.5, 0.9])))


def test_missing_pressure_raises():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 300)))
    with pytest.raises(ValueError, match="missing pressure"):
        pl.apply(pd.Series([0.1, np.nan, 0.9]))


def test_occupancy_report_counts_per_level():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 300)))
    rep = pl.occupancy(pd.Series(np.linspace(0, 1, 300)))
    assert all(rep[k] == pytest.approx(100, abs=5) for k in (1, 2, 3))
