"""D1's occlusion leg on a TWO-period match (review A-04, round 2).

``time_seconds`` is period-relative (ADR-017). The noise floor, the bridged-position RMSE and the simulated broadcast
occlusion each ordered a player's rows by that clock with no period key, so the two halves interleaved sample by
sample (measured: a 35x noise floor). Every helper must work within (game, period, player); each is held here to
per-period ground truth on a match whose halves differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scripts._coordination_occlusion import fov_mask, simulate_broadcast_occlusion  # noqa: E402
from scripts.derive_coordination_params import _bridge_rmse, _position_noise_rms  # noqa: E402
from tests.coordination._fixtures import make_coordination_match  # noqa: E402

_WIDTH_M = 40.0
_GAP_S = 0.4


def _two_halves() -> pd.DataFrame:
    """A 25 Hz, two-period match whose halves differ: period 2's players sit 30 m further up the pitch."""
    f = make_coordination_match(seconds=120.0, hz=25.0, provider="sportec", periods=2, noise_m=0.3)
    up = (f["period_id"] == 2).to_numpy() & ~f["is_ball"].to_numpy(bool)
    f.loc[up, "x"] = f.loc[up, "x"] + 30.0
    return f.reset_index(drop=True)


def _bridge_oracle(frames: pd.DataFrame, mask: np.ndarray, gap_s: float) -> float:
    """Independent per-(game, period, player) bridge RMSE: every masked run of at most ``gap_s`` with a detected row
    on both sides, linearly bridged from those two rows, against the true positions."""
    max_run = round(gap_s * float(frames["frame_rate"].iloc[0]))
    sq: list[float] = []
    players = frames.assign(_det=mask)[~frames["is_ball"].to_numpy(bool)]
    for _key, g in players.groupby(["game_id", "period_id", "player_id"], sort=True, observed=True):
        g = g.sort_values("time_seconds", kind="mergesort")
        det, x, y = g["_det"].to_numpy(), g["x"].to_numpy(float), g["y"].to_numpy(float)
        i = 0
        while i < len(g):
            if det[i]:
                i += 1
                continue
            j = i
            while j < len(g) and not det[j]:
                j += 1
            if j - i <= max_run and i > 0 and j < len(g):
                for k in range(i, j):
                    frac = (k - (i - 1)) / (j - (i - 1))
                    xe = x[i - 1] + (x[j] - x[i - 1]) * frac
                    ye = y[i - 1] + (y[j] - y[i - 1]) * frac
                    sq.append((xe - x[k]) ** 2 + (ye - y[k]) ** 2)
            i = j
    return float(np.sqrt(np.mean(sq)))


def test_fixture_preconditions():
    # ADR-032: two periods on one period-relative clock (the trap), halves that differ, and masked runs to bridge
    f = _two_halves()
    assert sorted(pd.unique(f["period_id"])) == [1, 2]
    p1, p2 = (f[(f["period_id"] == p).to_numpy() & ~f["is_ball"].to_numpy(bool)] for p in (1, 2))
    assert p1["time_seconds"].min() == p2["time_seconds"].min()  # the same clock restarts each period
    assert p2["x"].mean() - p1["x"].mean() > 25.0
    mask = fov_mask(f, width_m=_WIDTH_M)
    assert 0.2 < (~mask[~f["is_ball"].to_numpy(bool)]).mean() < 0.9


def test_noise_floor_is_computed_within_each_period():
    # A-04 + A-14: the floor is the median over per-(game, period, player, axis) residual-analysis intercepts, so the
    # +30 m jump at the period boundary never enters a player's run. Period-grouped it stays near the planted 0.3 m;
    # interleaving a player's two halves on the period-relative clock inflates it to metres.
    from silly_kicks.tracking.preprocess._butterworth import residual_analysis_noise_rms

    f = _two_halves()
    real = _position_noise_rms(f)
    per = [_position_noise_rms(f[(f["period_id"] == p).to_numpy()]) for p in (1, 2)]
    assert real < 1.0 and all(x < 1.0 for x in per)  # period-grouped: the +30 m step does not leak
    assert abs(real - float(np.median(per))) < 0.15  # the union median tracks the per-period medians (~0.3 m)
    # non-vacuity: ONE player's two halves, concatenated on the period-relative clock as a single series (the A-04
    # trap), carry the +30 m step and blow the residual-analysis floor past a metre.
    pid = f.loc[~f["is_ball"].to_numpy(bool), "player_id"].iloc[0]
    one = f[(f["player_id"] == pid) & ~f["is_ball"].to_numpy(bool)].sort_values("time_seconds")
    assert residual_analysis_noise_rms(one["x"].to_numpy(dtype=float), 25.0, np.arange(0.1, 5.0, 0.05)) > 1.0


def test_bridge_rmse_is_computed_within_each_period():
    f = _two_halves()
    mask = fov_mask(f, width_m=_WIDTH_M)
    assert _bridge_rmse(f, mask, _GAP_S) == pytest.approx(_bridge_oracle(f, mask, _GAP_S), rel=1e-12)


def test_occlusion_interpolates_within_each_period():
    f = _two_halves()
    both = simulate_broadcast_occlusion(f, width_m=_WIDTH_M)
    per = pd.concat(
        [simulate_broadcast_occlusion(f[(f["period_id"] == p).to_numpy()], width_m=_WIDTH_M) for p in (1, 2)]
    ).loc[f.index]
    pd.testing.assert_frame_equal(both[["x", "y"]], per[["x", "y"]])
    assert not both[["x", "y"]].equals(f[["x", "y"]])  # non-vacuity: masked rows really were filled


def test_occlusion_preserves_float32_storage_dtype():
    # ADR-106 (F1b): pining tracking frames store coords as float32. simulate_broadcast_occlusion writes
    # linearly-interpolated (float64) positions back into the coord columns; under pandas 3.0 a float64 write
    # into a float32 column raises LossySetitemError. The sim must cast to the stored dtype. Fixtures build
    # float64 frames, so only the full-corpus DGX run surfaced this -- hence this float32 guard.
    f = _two_halves()
    f["x"] = f["x"].astype(np.float32)
    f["y"] = f["y"].astype(np.float32)
    out = simulate_broadcast_occlusion(f, width_m=_WIDTH_M)  # must not raise
    assert out["x"].dtype == np.float32 and out["y"].dtype == np.float32  # storage dtype preserved
    assert not out[["x", "y"]].equals(f[["x", "y"]])  # non-vacuity: masked rows were filled
    # the stored values are the float64 interpolation cast to the stored dtype
    ref = simulate_broadcast_occlusion(
        f.assign(x=f["x"].astype(np.float64), y=f["y"].astype(np.float64)), width_m=_WIDTH_M
    )
    for axis in ("x", "y"):
        np.testing.assert_array_equal(out[axis].to_numpy(), ref[axis].to_numpy(dtype=np.float64).astype(np.float32))
