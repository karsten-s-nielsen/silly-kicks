"""Value snapshots for the cores that the tracking-batch cycle rewrote (owner ruling 2026-11-23, spec 2026-11-08
tracking-batch section 7).

When the frozen oracles ended (ruling C5), the preprocess, the two linkers, the shape-graph core and
``detect_line_breaking`` kept tests that check behaviour but pin no output values. Each case here pins the full
output of one public call on a seeded fixture (``_builders.py``), captured by ``capture_goldens.py``.

Float64 columns compare at ``rtol=1e-12`` (with ``atol=1e-12`` for values at or near zero). Float32-stored columns
(the preprocess outputs, ADR-106) compare within one float32 ulp: the float64 compute differs in its last bits between
CPUs, and the float32 rounding can then land on the neighbouring float32 (measured on arm64: one position in 2,132).
The velocity and acceleration columns are finite differences of that float32 position, so one position step moves
them by more than one ulp; they compare within an absolute bound per derivative (``DERIVED_FLOAT32_ABS``). Both rules
are owner rulings of 2026-11-23. So a real value change fails on every CPU while last-bit differences between CPUs
pass. NaN
placement must be equal. Every other column compares exactly, with the null kinds (``None``, ``NaN``, ``<NA>``)
treated as one. Dtypes are not compared: the parquet round trip and the pandas major version change them, and each
function's dtype contract has its own tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from regressions.tracking_batch._builders import CASES

GOLD = Path(__file__).resolve().parent
RTOL = 1e-12
ATOL = 1e-12
MAX_ULP_FLOAT32 = 1
#: Float32 preprocess columns that are finite differences of the float32 smoothed position. One position step (one
#: float32 ulp: at most 7.6e-6 m below 128 m) moves a velocity by at most frame rate x step (about 1.9e-4 m/s at
#: 25 Hz) and an acceleration by frame rate^2 x step (about 4.8e-3 m/s^2). Each bound is twice that worst case. Measured
#: on arm64: 5.8e-5 m/s and 1.2e-3 m/s^2 at most.
DERIVED_FLOAT32_ABS = {"vx": 5e-4, "vy": 5e-4, "speed": 5e-4, "accel_x": 1e-2, "accel_y": 1e-2, "accel": 1e-2}


def _as_objects(series: pd.Series) -> list:
    values = series.astype(object)
    return [None if pd.isna(v) else v for v in values.tolist()]


def assert_matches_golden(got: pd.DataFrame, golden: pd.DataFrame) -> None:
    assert list(got.columns) == list(golden.columns)
    assert len(got) == len(golden)
    for col in golden.columns:
        a, b = got[col], golden[col]
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            x, y = a.to_numpy(dtype="float64", na_value=np.nan), b.to_numpy(dtype="float64", na_value=np.nan)
            assert np.array_equal(np.isnan(x), np.isnan(y)), f"{col}: NaN placement differs"
            if b.dtype == np.float32 and col in DERIVED_FLOAT32_ABS:  # a derivative of the float32 position
                np.testing.assert_allclose(x, y, rtol=0, atol=DERIVED_FLOAT32_ABS[col], equal_nan=True, err_msg=col)
            elif b.dtype == np.float32:  # float32 storage: at most one float32 step between CPUs
                keep = ~np.isnan(y)
                try:
                    np.testing.assert_array_max_ulp(
                        x[keep].astype(np.float32), y[keep].astype(np.float32), maxulp=MAX_ULP_FLOAT32
                    )
                except AssertionError as exc:
                    raise AssertionError(f"{col}: {exc}") from None
            else:
                np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=col)
        else:
            assert _as_objects(a) == _as_objects(b), f"{col}: values differ"


@pytest.mark.parametrize("case", sorted(CASES))
def test_value_snapshot(case):
    golden = pd.read_parquet(GOLD / f"golden_{case}.parquet")
    got = CASES[case]()
    assert len(golden) > 0  # non-vacuity
    assert_matches_golden(got, golden)


def test_every_case_has_a_golden_and_no_golden_is_orphaned():
    on_disk = {p.name.removeprefix("golden_").removesuffix(".parquet") for p in GOLD.glob("golden_*.parquet")}
    assert on_disk == set(CASES)


def test_snapshot_compare_detects_a_value_change():
    # the compare is not vacuous: a change far above the tolerance fails, a last-bit change passes
    golden = CASES["shape_graph_edges"]()
    nudged = golden.assign(stability=golden["stability"] * (1 + 1e-15))
    assert_matches_golden(nudged, golden)
    moved = golden.assign(stability=golden["stability"] * (1 + 1e-9))
    with pytest.raises(AssertionError):
        assert_matches_golden(moved, golden)


def test_derived_float32_columns_compare_within_their_bound():
    # owner ruling 2026-11-23: a finite-difference column may move by its bound (one position step's effect, twice)
    golden = pd.DataFrame(
        {"vx": np.array([4.25, -1.5, 0.0], dtype=np.float32), "accel": np.array([2.0, -0.5, 9.0], dtype=np.float32)}
    )
    vx, accel = golden["vx"].to_numpy(), golden["accel"].to_numpy()
    inside = golden.assign(vx=vx + np.float32(4e-4), accel=accel - np.float32(9e-3))
    assert_matches_golden(inside, golden)
    with pytest.raises(AssertionError, match="vx"):
        assert_matches_golden(golden.assign(vx=vx + np.float32(7e-4)), golden)
    with pytest.raises(AssertionError, match="accel"):
        assert_matches_golden(golden.assign(accel=accel + np.float32(1.2e-2)), golden)


def test_float32_compare_allows_one_ulp_and_no_more():
    # owner ruling 2026-11-23: a float32-stored column may move by one float32 step (cross-CPU rounding), not two
    golden = pd.DataFrame({"x_smoothed": np.array([80.58667, 60.087456, 0.5], dtype=np.float32)})
    one_step = golden.assign(x_smoothed=np.nextafter(golden["x_smoothed"].to_numpy(), np.float32(np.inf)))
    assert_matches_golden(one_step, golden)
    two_steps = one_step.assign(x_smoothed=np.nextafter(one_step["x_smoothed"].to_numpy(), np.float32(np.inf)))
    with pytest.raises(AssertionError, match="x_smoothed"):
        assert_matches_golden(two_steps, golden)
