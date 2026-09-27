"""Task 4/5 gates: chunk_size / row-order / game-concatenation invariance of the DAS engine."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking._das_engine import compute_das
from silly_kicks.tracking._das_pack import pack_frames
from silly_kicks.tracking._das_params import DAS_PARAMS
from tests.tracking._das_golden import load_golden
from tests.tracking._das_helpers import run_engine

_ENGINES = ["numpy"]
try:
    import numba  # noqa: F401

    _ENGINES.append("numba")
except ImportError:
    pass


def _compute(frames, *, chunk_size, engine):
    packed = pack_frames(frames, attacking_direction_col="dir")
    return compute_das(packed, DAS_PARAMS, chunk_size=chunk_size, engine=engine)


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_chunk_size_is_byte_identical(engine, chunk_size):
    frames = load_golden().frames_for("S10")
    base = _compute(frames, chunk_size=10_000, engine=engine)
    got = _compute(frames, chunk_size=chunk_size, engine=engine)
    assert np.array_equal(base.team_das, got.team_das, equal_nan=True)
    assert np.array_equal(base.team_as, got.team_as, equal_nan=True)
    assert np.array_equal(base.player_das, got.player_das, equal_nan=True)
    assert np.array_equal(base.player_as, got.player_as, equal_nan=True)


@pytest.mark.parametrize("engine", _ENGINES)
def test_row_shuffle_invariant(engine):
    frames = load_golden().frames_for("S10")
    shuffled = frames.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a_team, a_player = run_engine(frames, DAS_PARAMS, engine=engine)
    b_team, b_player = run_engine(shuffled, DAS_PARAMS, engine=engine)
    a_team = a_team.sort_values(["game_id", "period_id", "frame_id"]).reset_index(drop=True)
    b_team = b_team.sort_values(["game_id", "period_id", "frame_id"]).reset_index(drop=True)
    np.testing.assert_array_equal(a_team["team_das"].to_numpy(), b_team["team_das"].to_numpy())
    a_player = a_player.sort_values(["game_id", "period_id", "frame_id", "player_id"]).reset_index(drop=True)
    b_player = b_player.sort_values(["game_id", "period_id", "frame_id", "player_id"]).reset_index(drop=True)
    np.testing.assert_array_equal(a_player["player_das"].to_numpy(), b_player["player_das"].to_numpy())


@pytest.mark.parametrize("engine", _ENGINES)
def test_concatenated_games_equal_per_game(engine):
    frames = load_golden().frames_for("S04")  # two games, disjoint frame ids
    whole_team, _ = run_engine(frames, DAS_PARAMS, engine=engine)
    whole_team = whole_team.sort_values(["game_id", "period_id", "frame_id"]).reset_index(drop=True)
    parts = []
    for _gid, sl in frames.groupby("game_id"):
        t, _ = run_engine(sl.reset_index(drop=True), DAS_PARAMS, engine=engine)
        parts.append(t)
    per_game = (
        __import__("pandas").concat(parts).sort_values(["game_id", "period_id", "frame_id"]).reset_index(drop=True)
    )
    np.testing.assert_array_equal(whole_team["team_das"].to_numpy(), per_game["team_das"].to_numpy())
