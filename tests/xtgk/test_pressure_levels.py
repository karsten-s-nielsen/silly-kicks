import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk._pressure_levels import (
    PressureLevels,
    band_of_zone,
    coalesce_frame_present_null_pressure,
)
from silly_kicks.xthreat._grid import N  # grid length (16)


def test_band_of_zone_deep_is_columns_0_and_1():
    # flat = (w-1-yj)*l + xi ; deep band = xi in {0,1}
    assert band_of_zone(0, N) == 0  # xi=0
    assert band_of_zone(1, N) == 0  # xi=1
    assert band_of_zone(2, N) == 1  # xi=2
    assert band_of_zone(N + 5, N) == 1  # xi=5 on the next row


def test_zone_conditional_terciles_are_within_band():
    # deep band globally LOW pressure (0..0.2); rest band HIGH (0.6..1.0). Global terciles would push
    # ALL deep actions into level 1; zone-conditional must give each band its own ~1/3-1/3-1/3.
    deep_p = np.linspace(0.0, 0.2, 150)
    rest_p = np.linspace(0.6, 1.0, 150)
    pressure = pd.Series(np.concatenate([deep_p, rest_p]))
    zones = np.concatenate([np.zeros(150, dtype=int), np.full(150, 5, dtype=int)])  # deep vs rest
    pl = PressureLevels(mode="zone_conditional").fit(pressure, zones=zones)
    lv = pl.apply(pressure, zones=zones)
    deep_lv, rest_lv = lv[:150], lv[150:]
    for sub in (deep_lv, rest_lv):
        assert set(np.unique(sub)) == {1, 2, 3}
        assert abs((sub == 3).sum() - 50) <= 3  # ~1/3 within band


def test_zone_conditional_apply_requires_zones():
    pressure = pd.Series(np.linspace(0.0, 1.0, 30))
    zones = np.tile([0, 5], 15)  # both bands populated so fit succeeds
    pl = PressureLevels(mode="zone_conditional").fit(pressure, zones=zones)
    with pytest.raises(ValueError, match="zones"):
        pl.apply(pressure)


def test_zone_conditional_meta_roundtrip():
    pressure = pd.Series(np.concatenate([np.linspace(0, 0.2, 60), np.linspace(0.6, 1.0, 60)]))
    zones = np.concatenate([np.zeros(60, dtype=int), np.full(60, 5, dtype=int)])
    pl = PressureLevels(mode="zone_conditional").fit(pressure, zones=zones)
    meta = pl.to_meta()
    assert meta["pressure_mode"] == "zone_conditional"
    pl2 = PressureLevels.from_meta(meta)
    assert np.array_equal(pl.apply(pressure, zones=zones), pl2.apply(pressure, zones=zones))


def test_global_meta_is_byte_identical_form():
    pl = PressureLevels(mode="global").fit(pd.Series(np.linspace(0, 1, 300)))
    meta = pl.to_meta()
    assert "pressure_mode" not in meta and "cutpoints" in meta  # SP1 on-disk form unchanged
    # absent pressure_mode => global back-compat
    pl2 = PressureLevels.from_meta({"cutpoints": meta["cutpoints"]})
    assert pl2.mode == "global"


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
