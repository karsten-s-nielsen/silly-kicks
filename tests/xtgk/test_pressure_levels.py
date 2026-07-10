import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk._pressure_levels import (
    PressureLevels,
    coalesce_frame_present_null_pressure,
)


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


# --- G8 frame-aware null-pressure rule (ADR-036 §5 amendment) ---


def test_coalesce_zeros_null_pressure_only_where_frame_present():
    # frame-present + null -> 0 (genuinely unpressured restart, LOW tercile); frame-absent + null
    # -> left null (genuine tracking gap, apply()'s fail-loud is the backstop); non-null untouched.
    pressure = pd.Series([0.5, np.nan, np.nan, 0.9])
    frame_present = pd.Series([True, True, False, False])
    out = coalesce_frame_present_null_pressure(pressure, frame_present)
    assert out.iloc[0] == 0.5  # non-null unchanged
    assert out.iloc[1] == 0.0  # frame present + null -> zero
    assert pd.isna(out.iloc[2])  # frame absent + null -> still null (backstop drops it)
    assert out.iloc[3] == 0.9


def test_coalesced_zero_pressure_lands_in_low_tercile():
    # an unpressured goal-kick (coalesced to 0) must map to tercile 1 under global cutpoints.
    pl = PressureLevels().fit(pd.Series([0.1, 0.5, 0.9] * 100))
    coalesced = coalesce_frame_present_null_pressure(pd.Series([np.nan]), pd.Series([True]))
    assert pl.apply(coalesced)[0] == 1


def test_coalesce_is_pure_no_mutation():
    pressure = pd.Series([np.nan, 0.3])
    frame_present = pd.Series([True, True])
    _ = coalesce_frame_present_null_pressure(pressure, frame_present)
    assert pd.isna(pressure.iloc[0])  # input Series not mutated
