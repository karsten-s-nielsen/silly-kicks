"""Structural perf guards (ADR-068): the per-game frame lookup in _off_ball_runs_kernel (TF-4) and
detect_off_ball_runs (TF-35) is built ONCE, not re-filtered per game.

The group_rows change is byte-identical, so the PARITY half is the (unchanged) correctness suite --
test_off_ball_runs.py + test_run_values_{detect,value}.py; this file adds only the STRUCTURAL half."""

import pandas as pd

import silly_kicks.tracking._off_ball_runs as _obr
import silly_kicks.tracking._run_values as _rv
from silly_kicks.tracking._off_ball_runs import _off_ball_runs_kernel
from silly_kicks.tracking._run_values import detect_off_ball_runs
from tests._perf_structural import call_counter
from tests.tracking.test_off_ball_runs import _make_action_at, _make_multi_frame_fixture

_N = 5
_PLAYERS = [
    {"player_id": 10, "team_id": 1, "positions": [(50, 34)] * _N},
    {"player_id": 11, "team_id": 1, "positions": [(50 + i * 5.0 / (_N - 1), 34) for i in range(_N)]},
    {"player_id": 12, "team_id": 1, "positions": [(30, 34 + i * 4.0 / (_N - 1)) for i in range(_N)]},
    {"player_id": 20, "team_id": 2, "positions": [(80, 34)] * _N},
    {"player_id": 21, "team_id": 2, "is_goalkeeper": True, "positions": [(102, 34)] * _N},
    {"player_id": 1, "team_id": 1, "is_goalkeeper": True, "positions": [(3, 34)] * _N},
]


def _two_game_fixture():
    frames = pd.concat(
        [
            _make_multi_frame_fixture(players=_PLAYERS, n_frames=_N, frame_rate=_N / 1.5, game_id=1),
            _make_multi_frame_fixture(players=_PLAYERS, n_frames=_N, frame_rate=_N / 1.5, game_id=2),
        ],
        ignore_index=True,
    )
    t = (_N - 1) * (1.5 / (_N - 1))
    actions = pd.concat(
        [
            _make_action_at(time_seconds=t, player_id=10, team_id=1, game_id=1, action_id=1),
            _make_action_at(time_seconds=t, player_id=10, team_id=1, game_id=2, action_id=2),
        ],
        ignore_index=True,
    )
    return actions, frames


def test_off_ball_runs_kernel_builds_lookup_once(monkeypatch):
    calls = call_counter(monkeypatch, _obr, "group_rows")
    actions, frames = _two_game_fixture()
    _off_ball_runs_kernel(actions, frames, home_team_id=1)
    assert calls["n"] == 1  # once total; pre-ADR-068 the whole `frames` table was re-filtered per game


def test_detect_off_ball_runs_builds_lookup_once(monkeypatch):
    calls = call_counter(monkeypatch, _rv, "group_rows")
    actions, frames = _two_game_fixture()
    detect_off_ball_runs(actions, frames)
    assert calls["n"] == 1
