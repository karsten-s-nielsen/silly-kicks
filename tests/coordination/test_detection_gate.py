"""A-08 (owner rulings 2026-10-04): ONE detection construct, tested by every family.

A side's detected share over a window is the fraction of its on-pitch player-samples that were RAW-detected (bridged
samples are not detected); a row's share is the minimum over its sides, emitted as ``coord_detected_share``, and the
row is ``insufficient_detection`` when that minimum is below its family's ``min_observed_fraction``. Before: four of the
eight families never tested the threshold, the other four tested each side's mean over the samples the metric read --
for a dyad player ~1.0 by construction (its runs are built from detected samples), and an empty window was labelled
``insufficient_detection`` even on a fully observed provider. The possession spectral row is exempt (no per-side
detection gate; D1 reports its occlusion error as the evidence).
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.coordination._catalog import PairSpec
from silly_kicks.coordination._columns import COORD_METHOD_FAMILIES
from silly_kicks.coordination._compute import compute_team_coordination, compute_vector_coding
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._detection import DetectionCounts, insufficient_detection, row_detected_share
from silly_kicks.coordination._report import CoordinationCoverageWarning
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows
from silly_kicks.id_compat import same_id
from tests.coordination._fixtures import make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore:vx/vy columns not found")

_TEAM_A = 1
_HIDDEN = (101, 102, 103, 104)  # 4 of team A's 10 outfield players, never detected: team A's share is exactly 0.6
_SOURCE = {
    "relative_phase": "coord_rp_source",
    "cross_correlation": "coord_xc_source",
    "vector_coding": "coord_vc_source",
    "coherence": "coord_coh_source",
}


# --------------------------------------------------------------------------- the construct, as a unit
def test_share_counts_detected_over_on_pitch_player_samples():
    on = np.array([[1, 1], [1, 1], [1, 0], [0, 0]], dtype=bool)  # 4 samples x 2 players
    det = np.array([[1, 0], [1, 1], [0, 0], [0, 0]], dtype=bool)
    c = DetectionCounts.from_masks(on, det)
    assert c.share(0, 4) == pytest.approx(3 / 5)  # 3 detected of 5 on-pitch player-samples
    assert c.share(2, 3) == 0.0  # on the pitch, never detected
    assert math.isnan(c.share(3, 4))  # nobody on the pitch: no share
    assert math.isnan(c.share(1, 1))  # an empty window


def test_masks_must_agree():
    with pytest.raises(ValueError, match="shape"):
        DetectionCounts.from_masks(np.ones(3, bool), np.ones(4, bool))
    with pytest.raises(ValueError, match="on the pitch"):
        DetectionCounts.from_masks(np.array([True, False]), np.array([True, True]))


def test_row_share_is_the_minimum_over_sides_ignoring_an_absent_side():
    on = np.ones(4, bool)
    full = DetectionCounts.from_masks(on, on)
    half = DetectionCounts.from_masks(on, np.array([1, 1, 0, 0], bool))
    absent = DetectionCounts.from_masks(np.zeros(4, bool), np.zeros(4, bool))
    assert row_detected_share([full, half], 0, 4) == 0.5
    assert row_detected_share([full, absent], 0, 4) == 1.0  # a side never on the pitch has no say
    assert math.isnan(row_detected_share([absent, absent], 0, 4))


def test_gate_both_sides_of_the_threshold():
    assert insufficient_detection(np.nextafter(0.6, 0.0), 0.6)
    assert not insufficient_detection(0.6, 0.6)
    assert not insufficient_detection(float("nan"), 1.0)  # no share -> no detection verdict


# --------------------------------------------------------------------------- every family tests it
def _params(threshold: float = 0.0, family: str | None = None) -> CoordinationParams:
    fam = {k: (threshold if family in (None, k) else 0.0) for k in COORD_METHOD_FAMILIES}
    return dataclasses.replace(CoordinationParams(), n_surrogates=0, welch_segment_s=20.0, min_observed_fraction=fam)


def _match(provider: str = "skillcorner", seconds: float = 300.0, **kw) -> pd.DataFrame:
    return make_coordination_match(
        seconds=seconds, hz=10.0, provider=provider, oscillation_cpm=0.5, phase_offset_deg=40.0, **kw
    )


def _undetect(f: pd.DataFrame, pids, lo: float = -np.inf, hi: float = np.inf) -> pd.DataFrame:
    f = f.copy()
    t = f["time_seconds"].to_numpy(dtype=np.float64)
    rows = f["player_id"].isin(list(pids)).fillna(False).to_numpy(dtype=bool) & (t >= lo) & (t < hi)
    vis = f["visibility"].astype(object).to_numpy(copy=True)
    vis[rows] = False
    f["visibility"] = vis
    return f


def _run(frames: pd.DataFrame, params: CoordinationParams, windows: pd.DataFrame | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        return compute_team_coordination(
            frames, params=params, windows=windows if windows is not None else period_windows(frames)
        )


def _is(series: pd.Series, team) -> np.ndarray:
    return series.map(lambda v: same_id(v, team)).to_numpy(dtype=bool)


def _in(series: pd.Series, ids) -> np.ndarray:
    """Output ids are canonical (ADR-019): match through id_compat, never by value."""
    return series.map(lambda v: any(same_id(v, i) for i in ids)).to_numpy(dtype=bool)


def _family_rows(res, family: str) -> tuple[pd.Series, pd.Series]:
    """(token, share) of the family's team-level rows that involve team A on the period window."""
    if family in _SOURCE:
        df = res.pair[(res.pair.window_kind == "period") & (res.pair.level == "team_team")]
        return df[_SOURCE[family]], df["coord_detected_share"]
    if family == "spectral":
        df = res.spectral[(res.spectral.signal != "possession") & _is(res.spectral.team_id, _TEAM_A)]
        return df["coord_spectral_source"], df["coord_detected_share"]
    if family == "cluster":
        t = res.cluster_team[_is(res.cluster_team.team_id, _TEAM_A)]
        p = res.cluster_player[_is(res.cluster_player.team_id, _TEAM_A) & ~_in(res.cluster_player.player_id, _HIDDEN)]
        tokens = pd.concat([t["coord_cluster_source"], p["coord_cluster_player_source"]], ignore_index=True)
        return tokens, pd.concat([t["coord_detected_share"], p["coord_detected_share"]], ignore_index=True)
    df = getattr(res, family)  # team_sync, rsi
    return df[f"coord_{family}_source"], df["coord_detected_share"]


@pytest.fixture(scope="module")
def team_a_at_sixty_percent():
    return _undetect(_match(), _HIDDEN)


@pytest.mark.parametrize("family", COORD_METHOD_FAMILIES)
def test_every_family_gates_on_the_row_share_both_sides(family, team_a_at_sixty_percent):
    # team A's share is exactly 6/10 = 0.6 and team B's 1.0: every team-level row involving A has share 0.6. At a
    # threshold of exactly 0.6 nothing is flagged (0.6 is not below 0.6); one ulp above, every such row is.
    for threshold, flagged in ((0.6, False), (np.nextafter(0.6, 1.0), True)):
        tokens, shares = _family_rows(_run(team_a_at_sixty_percent, _params(threshold, family)), family)
        assert len(tokens) > 0, family  # non-vacuity
        assert (shares.astype(float) == 0.6).all(), (family, sorted(set(shares)))
        assert ((tokens == "insufficient_detection").to_numpy() == flagged).all(), (family, threshold, set(tokens))


def test_a_cluster_player_tests_the_lower_of_its_team_and_its_own_share(team_a_at_sixty_percent):
    # owner ruling: a cluster player row tests min(team share, player share) -- a never-detected player fails even
    # where its team passes
    res = _run(team_a_at_sixty_percent, _params(0.5, "cluster"))
    team = res.cluster_team[_is(res.cluster_team.team_id, _TEAM_A)]
    assert (team["coord_cluster_source"] != "insufficient_detection").all()
    hidden = res.cluster_player[_in(res.cluster_player.player_id, _HIDDEN)]
    assert len(hidden) == 2 * len(_HIDDEN)  # both axes, every hidden player
    assert (hidden["coord_detected_share"].astype(float) == 0.0).all()
    assert (hidden["coord_cluster_player_source"] == "insufficient_detection").all()


def test_a_dyad_player_off_camera_most_of_the_window_fails():
    # the non-vacuity case of the ruling: player 105 is detected only in the first 20% of the 300-s period, so every
    # dyad holding it has share 0.2 < 0.5 -- while its own run (the first 60 s) is perfectly scoreable: the former
    # per-side mean over the samples the metric read was 1.0, and the gate could never fail
    f = _undetect(_match(), [105], lo=60.0)
    res = _run(f, _params(0.5))
    dyads = res.pair[(res.pair.level == "dyad") & (_in(res.pair.player_a_id, [105]) | _in(res.pair.player_b_id, [105]))]
    assert len(dyads) > 0
    assert np.allclose(dyads["coord_detected_share"].astype(float), 0.2)
    assert (dyads["coord_rp_source"] == "insufficient_detection").all()
    assert (dyads["coord_n_samples"].astype(float) >= 3).all()  # the run IS there: only the share fails it
    side_105 = np.where(_in(dyads.player_a_id, [105]), dyads.coord_observed_fraction_a, dyads.coord_observed_fraction_b)
    assert np.allclose(side_105.astype(float), 0.2)  # the coverage column reports the same raw-detection share


def test_a_fully_observed_provider_is_never_insufficient_detection():
    res = _run(_match(provider="sportec"), _params(1.0))  # the strictest threshold there is
    for table, cols in (
        ("pair", list(_SOURCE.values())),
        ("pair_phase", ["coord_rp_source", "coord_vc_source"]),
        ("spectral", ["coord_spectral_source"]),
        ("cluster_team", ["coord_cluster_source"]),
        ("cluster_player", ["coord_cluster_player_source"]),
        ("team_sync", ["coord_team_sync_source"]),
        ("rsi", ["coord_rsi_source"]),
    ):
        df = getattr(res, table)
        for c in cols:
            assert not (df[c] == "insufficient_detection").any(), (table, c)
        gated = df["coord_detected_share"].dropna().astype(float)
        assert (gated == 1.0).all(), table


def test_an_empty_window_is_too_short_not_insufficient_detection():
    # a window inside a 60-s dead-ball stoppage: both teams ARE on the pitch (share 1.0) but no segment sample is in
    # the window -- too_short, not insufficient_detection (the former gate failed every empty window)
    f = _match(provider="sportec", dead_intervals=[(100.0, 160.0)])
    w = period_windows(f, length_s=40.0, step_s=40.0)
    w = w[(w.start_time_s >= 100.0) & (w.end_time_s <= 160.0)].reset_index(drop=True)
    assert len(w) == 1  # fixture precondition: the [120, 160) window sits inside the stoppage
    res = _run(f, _params(0.5), windows=w)
    team = res.pair[res.pair.level == "team_team"]
    assert (team["coord_detected_share"].astype(float) == 1.0).all()
    assert (team["coord_rp_source"] == "too_short").all()


def test_a_side_on_the_pitch_but_never_detected_is_insufficient_detection():
    # the other side: team A entirely undetected inside the window -> share 0.0 -> insufficient_detection
    f = _undetect(_match(), range(100, 111), lo=120.0, hi=160.0)
    w = period_windows(f, length_s=40.0, step_s=40.0)
    w = w[(w.start_time_s == 120.0)].reset_index(drop=True)
    res = _run(f, _params(0.5), windows=w)
    team = res.pair[res.pair.level == "team_team"]
    assert len(team) > 0
    assert (team["coord_detected_share"].astype(float) == 0.0).all()
    assert (team["coord_rp_source"] == "insufficient_detection").all()


def test_insufficient_detection_precedes_insufficient_players_and_not_commensurate():
    # Task 14 precedence: goal_end_unresolved, insufficient_detection, insufficient_players, too_short, ...,
    # not_commensurate -- a 3-player team that is also never detected is insufficient_detection first
    small = _undetect(_match(n_outfield=3, with_gk=False), [101, 102, 103])
    res = _run(small, _params(0.5))
    team_a = res.cluster_team[_is(res.cluster_team.team_id, _TEAM_A)]
    assert (team_a["coord_cluster_source"] == "insufficient_detection").all()
    hidden = _undetect(_match(), range(101, 111))
    sig = build_coordination_signals(hidden, windows=period_windows(hidden), params=_params(0.5))
    area_vs_spread = PairSpec("team_team", "convex_hull_area", "spread", "canonical")
    vc, _phase, _rep = compute_vector_coding(sig, pairs=[area_vs_spread])
    assert len(vc) > 0  # m^2 vs m: not commensurate, but the detection gate comes first
    assert (vc["coord_vc_source"] == "insufficient_detection").all()


def test_the_possession_spectral_row_is_exempt():
    # exempt with its reason (owner ruling 2026-10-04): the row carries no per-side detection gate; D1 reports its
    # occlusion error as the evidence. Even with team A never detected and the strictest threshold, it is never
    # insufficient_detection and carries no share.
    f = _undetect(_match(), range(100, 111))
    res = _run(f, _params(1.0))
    poss = res.spectral[res.spectral.signal == "possession"]
    assert len(poss) > 0
    assert (poss["coord_spectral_source"] != "insufficient_detection").all()
    assert poss["coord_detected_share"].isna().all()
    team_rows = res.spectral[res.spectral.signal != "possession"]
    assert (team_rows["coord_spectral_source"] == "insufficient_detection").any()  # the gate itself is live
