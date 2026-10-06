"""TF-58 Task 17: raw coordination time-series primitive."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import scipy.stats

from silly_kicks.coordination._catalog import PairSpec
from silly_kicks.coordination._compute import compute_relative_phase
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._series import compute_coordination_series
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows
from tests.coordination._fixtures import make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore:vx/vy columns not found")
_FAST = dataclasses.replace(CoordinationParams(), n_surrogates=0, welch_segment_s=20.0)


@pytest.fixture(scope="module")
def signals():
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", phase_offset_deg=40.0)
    return build_coordination_signals(f, windows=period_windows(f), params=_FAST)


def test_relative_phase_series_matches_pair_statistics(signals):
    pair, _phase, _rep = compute_relative_phase(signals)
    row = pair[(pair.level == "team_team") & (pair.signal_a == "centroid_x")].iloc[0]
    series = compute_coordination_series(signals, kind="relative_phase")
    sub = series[(series.level == "team_team") & (series.signal_a == "centroid_x")]
    circ_mean = np.degrees(
        scipy.stats.circmean(np.radians(sub.value.to_numpy()), high=np.pi, low=-np.pi)  # type: ignore[arg-type]
    )
    assert abs(np.degrees(np.angle(np.exp(1j * np.radians(circ_mean - float(row.coord_rp_mean_deg)))))) < 1.0


def test_coupling_angle_series_omits_stationary(signals):
    series = compute_coordination_series(signals, kind="coupling_angle")
    assert len(series)
    assert series.value.between(0.0, 360.0).all()


def test_coupling_angle_series_only_for_vector_coding_eligible_pairs(signals):
    # A-51: coupling angle is vector coding's output, so the series must emit it ONLY for pairs VC would score.
    # dyad has no vector_coding method (METHODS_BY_LEVEL), yet relative phase does -- the old relative-phase filter
    # emitted ~1.1M dyad coupling angles VC never computes.
    series = compute_coordination_series(signals, kind="coupling_angle")
    assert "dyad" not in set(series["level"]), "VC does not run on dyads; the series must not either"


def test_coupling_angle_series_refuses_non_commensurate_pairs(signals):
    # A-51: VC refuses non-commensurate pairs (_compute.py not_commensurate). The series must mirror that -- but
    # relative phase, which has no commensurate requirement, must still emit the pair (the restriction is VC-specific).
    commensurate = PairSpec("team_team", "centroid_x", "centroid_x", "canonical")
    non_commensurate = PairSpec("team_team", "convex_hull_area", "spread", "canonical")  # m^2 vs m
    assert not non_commensurate.commensurate
    pairs = [commensurate, non_commensurate]

    coupling = compute_coordination_series(signals, kind="coupling_angle", pairs=pairs)
    emitted = set(zip(coupling["signal_a"], coupling["signal_b"], strict=True))
    assert ("centroid_x", "centroid_x") in emitted  # non-vacuity: the commensurate pair still scores
    assert ("convex_hull_area", "spread") not in emitted

    rp = compute_coordination_series(signals, kind="relative_phase", pairs=pairs)
    rp_emitted = set(zip(rp["signal_a"], rp["signal_b"], strict=True))
    assert ("convex_hull_area", "spread") in rp_emitted  # relative phase is not commensurate-gated


def test_cluster_amplitude_is_segment_level(signals):
    series = compute_coordination_series(signals, kind="cluster_amplitude")
    assert set(series.columns) == {"game_id", "period_id", "segment_id", "time_s", "kind", "team_id", "axis", "value"}
    assert (series["kind"] == "cluster_amplitude").all()
    assert series["segment_id"].ge(0).all()
    assert series.value.between(0.0, 1.0 + 1e-9).all()


def test_long_table_columns(signals):
    rp = compute_coordination_series(signals, kind="relative_phase")
    assert list(rp.columns) == [
        "game_id",
        "period_id",
        "segment_id",
        "time_s",
        "kind",
        "level",
        "signal_a",
        "signal_b",
        "axis",
        "team_a_id",
        "team_b_id",
        "player_a_id",
        "player_b_id",
        "value",
    ]
