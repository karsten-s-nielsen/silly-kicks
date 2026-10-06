"""Child rows of degraded parents are kept, tokenised (spec 7.13 / ADR-042; owner ruling 2026-10-04; A-21).

A degraded cluster-team window used to drop ALL its player rows, and a degraded pair row emitted NO pair-phase rows:
the child tables lost rows silently, so a consumer could not tell "degraded" from "never computed". Every parent now
emits what a healthy parent would -- one player row per roster member, the window's ``n_phases`` phase rows (none on
an NA window) -- with real keys, NaN metrics and the PARENT's token, whatever the reason (reason-independent). A
healthy parent's own degraded child keeps its own token. A-21: a cluster player with fewer than
``MIN_CLUSTER_SAMPLES`` usable samples is ``too_short`` (its rho_k was exactly 1, its phase SD 0).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from silly_kicks.coordination._columns import (
    COORD_METHOD_FAMILIES,
    COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS,
    COORDINATION_PAIR_KEYS,
    COORDINATION_PAIR_PHASE_METRIC_COLUMNS,
)
from silly_kicks.coordination._compute import (
    MIN_CLUSTER_SAMPLES,
    compute_cluster_phase,
    compute_relative_phase,
    compute_vector_coding,
)
from silly_kicks.coordination._config import CoordinationParams
from silly_kicks.coordination._report import CoordinationCoverageWarning
from silly_kicks.coordination._signals import build_coordination_signals
from silly_kicks.coordination._windows import period_windows
from silly_kicks.id_compat import same_id
from tests.coordination._fixtures import make_coordination_match

pytestmark = [
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
]
_PLAYER_METRICS = [c for c in COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS if c != "coord_detected_share"]
_PHASE_METRICS = [c for c in COORDINATION_PAIR_PHASE_METRIC_COLUMNS if c != "coord_detected_share"]


class _NoGoal:
    def get(self, *_a):  # duck-typed GoalMap: every defended end unresolved
        return None


def _params(threshold: float = 0.0) -> CoordinationParams:
    return dataclasses.replace(
        CoordinationParams(),
        n_surrogates=0,
        welch_segment_s=20.0,
        min_observed_fraction=dict.fromkeys(COORD_METHOD_FAMILIES, threshold),
    )


def _match(provider: str = "sportec", seconds: float = 120.0, **kw) -> pd.DataFrame:
    return make_coordination_match(
        seconds=seconds, hz=10.0, provider=provider, oscillation_cpm=0.5, phase_offset_deg=40.0, **kw
    )


def _hide_team(f: pd.DataFrame, team) -> pd.DataFrame:
    f = f.copy()
    rows = f["team_id"].map(lambda v: same_id(v, team)).to_numpy(dtype=bool) & ~f["is_ball"].to_numpy(dtype=bool)
    vis = f["visibility"].astype(object).to_numpy(copy=True)
    vis[rows] = False
    f["visibility"] = vis
    return f


def _caller(frames: pd.DataFrame, spans) -> pd.DataFrame:
    """Caller windows ``(window_kind, start_s, end_s, n_phases)`` over the fixture's period."""
    base = period_windows(frames).iloc[[0]]
    rows, next_id = [], {}
    for kind, lo, hi, n in spans:
        r = base.copy()
        r["window_source"] = "caller"
        r["window_kind"] = kind
        r["window_id"] = pd.array([next_id.get(kind, 0)], dtype="Int64")
        next_id[kind] = next_id.get(kind, 0) + 1
        r["start_time_s"], r["end_time_s"] = float(lo), float(hi)
        r["n_phases"] = pd.array([n], dtype="Int64")
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def _signals(frames, windows, params=None, **kw):
    return build_coordination_signals(frames, windows=windows, params=params or _params(), **kw)


# --------------------------------------------------------------------------- cluster players
def _cluster(sig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoordinationCoverageWarning)
        team, players, _sync, _rep = compute_cluster_phase(sig)
    return team, players


def _children_of(team_row: pd.Series, players: pd.DataFrame) -> pd.DataFrame:
    keys = ("window_kind", "window_id", "axis")
    sel = np.ones(len(players), dtype=bool)
    for k in keys:
        sel &= (players[k] == team_row[k]).to_numpy(dtype=bool)
    sel &= players["team_id"].map(lambda v: same_id(v, team_row["team_id"])).to_numpy(dtype=bool)
    return players[sel]


def _assert_tokenised_children(team: pd.DataFrame, players: pd.DataFrame, token: str, roster: int) -> None:
    degraded = team[team["coord_cluster_source"] == token]
    assert len(degraded) > 0, token  # non-vacuity: the token really is reached
    for _, t_row in degraded.iterrows():
        kids = _children_of(t_row, players)
        assert len(kids) == roster, (token, len(kids), roster)
        assert (kids["coord_cluster_player_source"] == token).all(), token
        assert kids[_PLAYER_METRICS].isna().all().all(), token
        assert kids["player_id"].notna().all() and kids["team_id"].notna().all()  # real keys, joinable


def test_a_healthy_cluster_window_emits_one_player_row_per_roster_member():
    # the count every degraded parent must match: 11 players (10 outfield + keeper, Duarte) per team and axis
    f = _match()
    team, players = _cluster(_signals(f, period_windows(f)))
    healthy = team[team["coord_cluster_source"] == "scored"]
    assert len(healthy) > 0
    for _, t_row in healthy.iterrows():
        kids = _children_of(t_row, players)
        assert len(kids) == 11
        assert kids["coord_rho_k"].notna().any()


def test_an_insufficient_detection_cluster_window_keeps_its_player_rows():
    f = _hide_team(_match(provider="skillcorner"), 1)
    team, players = _cluster(_signals(f, period_windows(f), params=_params(0.5)))
    _assert_tokenised_children(team, players, "insufficient_detection", roster=11)


def test_an_insufficient_players_cluster_window_keeps_its_player_rows():
    f = _match(n_outfield=3, with_gk=False)  # 3 < min_players: no usable sample
    team, players = _cluster(_signals(f, period_windows(f)))
    _assert_tokenised_children(team, players, "insufficient_players", roster=3)


def test_a_too_short_cluster_window_keeps_its_player_rows():
    f = _match()
    team, players = _cluster(_signals(f, _caller(f, [("sliding", 50.0, 50.1, pd.NA)])))  # one sample
    _assert_tokenised_children(team, players, "too_short", roster=11)


def test_a_cluster_player_with_one_usable_sample_is_too_short_both_sides():
    # A-21: a substitute entering at 60 s has ONE usable sample in [50, 60.1) -- its window-mean relative phasor IS
    # that sample, so rho_k was exactly 1 and the phase SD 0. Two samples ([50, 60.2)) are scored.
    f = _match(substitution=(60.0, 3))
    sub = 1003  # team 1's slot-3 substitute (fixture id 100 * team + 900 + slot)
    assert MIN_CLUSTER_SAMPLES == 2
    for end, expect in ((60.1, "too_short"), (60.2, "scored")):
        _team, players = _cluster(_signals(f, _caller(f, [("sliding", 50.0, end, pd.NA)])))
        row = players[players["player_id"].map(lambda v: same_id(v, sub)).to_numpy(dtype=bool)]
        assert len(row) == 2, end  # both axes
        assert (row["coord_cluster_player_source"] == expect).all(), (end, set(row["coord_cluster_player_source"]))
        if expect == "too_short":
            assert row[["coord_rho_k", "coord_phi_sd_deg", "coord_phi_mean_deg"]].isna().all().all()
        else:
            assert row["coord_rho_k"].notna().all()


# --------------------------------------------------------------------------- pair-phase rows
def _rp(sig):
    pair, phase, _rep = compute_relative_phase(sig)
    return pair, phase


def _phase_children(pair_row: pd.Series, phase: pd.DataFrame) -> pd.DataFrame:
    sel = np.ones(len(phase), dtype=bool)
    for k in COORDINATION_PAIR_KEYS:
        a, b = phase[k], pair_row[k]
        sel &= a.isna().to_numpy() if pd.isna(b) else a.map(lambda v, b=b: same_id(v, b)).to_numpy(dtype=bool)
    return phase[sel]


@pytest.mark.parametrize(
    ("token", "build"),
    [
        (
            "insufficient_detection",
            lambda: (lambda f: _signals(f, _caller(f, [("possession", 0.0, 60.0, 3)]), params=_params(0.5)))(
                _hide_team(_match(provider="skillcorner"), 1)
            ),
        ),
        (
            "goal_end_unresolved",
            lambda: (lambda f: _signals(f, _caller(f, [("possession", 0.0, 60.0, 3)]), goal_map=_NoGoal()))(_match()),
        ),
        ("too_short", lambda: (lambda f: _signals(f, _caller(f, [("possession", 50.0, 50.2, 3)])))(_match())),
        ("no_possession_role", lambda: (lambda f: _signals(f, _caller(f, [("sliding", 0.0, 60.0, 3)])))(_match())),
    ],
)
def test_a_degraded_pair_row_keeps_its_windows_phase_rows(token, build):
    pair, phase = _rp(build())
    degraded = pair[pair["coord_rp_source"] == token]
    assert len(degraded) > 0, token  # non-vacuity: the token really is reached
    for _, p_row in degraded.iterrows():
        kids = _phase_children(p_row, phase)
        assert sorted(kids["phase_index"].astype(int)) == [1, 2, 3], token
        assert (kids["coord_rp_source"] == token).all(), token
        assert kids[_PHASE_METRICS].isna().all().all(), token


def test_a_degraded_pair_row_on_an_na_window_has_no_phase_rows():
    f = _hide_team(_match(provider="skillcorner"), 1)
    pair, phase = _rp(_signals(f, _caller(f, [("possession", 0.0, 60.0, pd.NA)]), params=_params(0.5)))
    assert (pair["coord_rp_source"] == "insufficient_detection").any()  # degraded parents exist
    assert len(phase) == 0  # a healthy parent on an NA window has none either


@pytest.mark.parametrize("compute", [compute_relative_phase, compute_vector_coding])
def test_pair_phase_rows_are_conserved_against_the_window_n_phases(compute):
    # every pair row, scored or degraded, carries exactly its window's n_phases phase rows (the mutant that drops a
    # degraded parent's children fails here)
    f = _hide_team(_match(provider="skillcorner"), 1)
    w = _caller(f, [("possession", 0.0, 40.0, 3), ("possession", 40.0, 80.0, 2), ("possession", 80.0, 120.0, pd.NA)])
    sig = _signals(f, w, params=_params(0.5))
    pair, phase, _rep = compute(sig)
    n_of = {
        int(wid): (0 if pd.isna(nph) else int(nph))
        for wid, nph in zip(w["window_id"].to_numpy(), w["n_phases"].to_numpy(), strict=True)
    }
    expected = sum(n_of[int(wid)] for wid in pair["window_id"])
    assert len(phase) == expected
    assert (pair[pair.columns[pair.columns.str.endswith("_source")][0]] != "scored").any()  # degraded parents count
