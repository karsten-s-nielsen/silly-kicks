"""Review A-19: the report's coverage sample counters are POPULATED, not a made-up 0 (spec 7.13, ADR-042).

``samples_unobserved`` (team samples voided because an on-pitch player has no bridged position), ``samples_stationary``
(vector-coding consecutive-diff samples omitted as stationary) and ``samples_below_min_players`` (cluster samples with
the team present but fewer than ``min_players`` valid phasors) were declared, initialised and never incremented. Each is
pinned here from BOTH sides: it goes positive when its condition occurs and stays 0 when it does not.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from silly_kicks.coordination import CoordinationParams, _signals, compute_team_coordination
from tests.coordination._fixtures import make_coordination_match

pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning")


def _clean(**over):
    return dataclasses.replace(CoordinationParams(), n_surrogates=0, welch_segment_s=20.0, **over)


def test_samples_below_min_players_is_populated_under_occlusion_and_zero_when_clean():
    # occlusion: hide 8 of one team's outfielders over a stretch -> many cluster samples have < min_players valid.
    f = make_coordination_match(seconds=300.0, hz=10.0, provider="skillcorner", phase_offset_deg=40.0)
    outfield = [p for p in f.loc[~f["is_ball"], "player_id"].dropna().unique()][:8]
    stretch = (f["time_seconds"] > 100.0).to_numpy() & (f["time_seconds"] < 150.0).to_numpy()
    f.loc[f["player_id"].isin(outfield).to_numpy(na_value=False) & stretch, "visibility"] = False
    occluded = compute_team_coordination(
        f, params=dataclasses.replace(CoordinationParams.for_provider("skillcorner"), n_surrogates=0)
    ).report
    assert occluded.samples_below_min_players > 0
    assert occluded.conservation_errors() == []

    clean = compute_team_coordination(
        make_coordination_match(seconds=300.0, hz=10.0, provider="sportec", phase_offset_deg=40.0), params=_clean()
    ).report
    assert clean.samples_below_min_players == 0  # every team sample has the full roster


def test_samples_stationary_is_populated_when_vector_coding_omits_stationary_diffs():
    # Vector coding omits a consecutive-diff sample when BOTH signals are within their epsilon. A huge vc_epsilon makes
    # every diff stationary (deterministic); the default epsilon on the same moving signals omits none.
    f = make_coordination_match(seconds=150.0, hz=10.0, provider="sportec", oscillation_cpm=0.5)
    big_eps = {k: 1e6 for k in CoordinationParams().vc_epsilon}
    stationary = compute_team_coordination(f, params=_clean(vc_epsilon=big_eps)).report
    assert stationary.samples_stationary > 0
    # the "omits none" leg pins vc_epsilon to zero: the mechanism is "within-epsilon -> stationary", and the
    # commit-2 derivation default vc_epsilon is non-zero (~0.0015), so the default no longer omits exactly none.
    zero_eps = {k: 0.0 for k in CoordinationParams().vc_epsilon}
    assert compute_team_coordination(f, params=_clean(vc_epsilon=zero_eps)).report.samples_stationary == 0


def _player(x, y, grid_n, *, nan_slice=None, gk=False):
    xa = np.full(grid_n, float(x))
    ya = np.full(grid_n, float(y))
    if nan_slice is not None:
        xa[nan_slice] = np.nan
        ya[nan_slice] = np.nan
    on_pitch = np.ones(grid_n, dtype=bool)
    return _signals._PlayerXY(xa, ya, [(0, grid_n)], on_pitch, on_pitch.copy(), gk)


def test_samples_unobserved_counts_on_pitch_players_with_no_valid_position():
    # A team sample is unobserved when an ON-PITCH player's resampled position is NaN. Built directly on `_team_signals`
    # because A-02 (best-estimate positions) + bridging make the condition rare in the full pipeline. One of eleven
    # on-pitch players has NaN positions over [50, 60): those ten samples are voided and counted.
    grid_n, team = 100, 1
    xs = np.linspace(20.0, 90.0, 11)
    ys = np.linspace(5.0, 63.0, 11)
    players = {i: _player(xs[i], ys[i], grid_n) for i in range(1, 11)}
    players[0] = _player(xs[0], ys[0], grid_n, nan_slice=slice(50, 60))  # on-pitch but no position for 10 samples
    team_xy = {team: players}
    counters = {"samples_unobserved": 0}
    _signals._team_signals(
        team_xy,
        team,
        2,
        {team: 0.0},
        10.0,
        grid_n,
        None,
        CoordinationParams(),
        np.zeros(grid_n, dtype=bool),
        counters=counters,
    )
    assert counters["samples_unobserved"] == 10

    # the other side: every on-pitch player has a position -> nothing is unobserved
    counters_ok = {"samples_unobserved": 0}
    _signals._team_signals(
        {team: {i: _player(xs[i], ys[i], grid_n) for i in range(11)}},
        team,
        2,
        {team: 0.0},
        10.0,
        grid_n,
        None,
        CoordinationParams(),
        np.zeros(grid_n, dtype=bool),
        counters=counters_ok,
    )
    assert counters_ok["samples_unobserved"] == 0


def test_a_merged_report_sums_every_sample_counter():
    # the counters sum across shard reports (conservation under merge, ADR-042).
    f = make_coordination_match(seconds=150.0, hz=10.0, provider="sportec", oscillation_cpm=0.5)
    big_eps = {k: 1e6 for k in CoordinationParams().vc_epsilon}
    one = compute_team_coordination(f, params=_clean(vc_epsilon=big_eps)).report
    merged = one.merge(one)
    assert merged.samples_stationary == 2 * one.samples_stationary
    assert merged.samples_below_min_players == 2 * one.samples_below_min_players
    assert merged.samples_unobserved == 2 * one.samples_unobserved
