"""VAEP windowing e2e tests using StatsBomb WC2018 H5 fixture (TF-29)."""

from pathlib import Path

import pandas as pd
import pytest

import silly_kicks.vaep.labels as lab
from silly_kicks.spadl import add_names
from silly_kicks.spadl.utils import add_possessions

_H5_PATH = Path(__file__).resolve().parent.parent / "datasets" / "statsbomb" / "spadl-WorldCup-2018.h5"


@pytest.fixture
def wc2018_actions() -> pd.DataFrame:
    """Load a single WC2018 match's SPADL actions from the H5 fixture."""
    if not _H5_PATH.exists():
        pytest.skip(f"H5 fixture not found: {_H5_PATH}")
    games = pd.read_hdf(_H5_PATH, "games")
    game_id = games.iloc[0]["game_id"]
    actions = pd.read_hdf(_H5_PATH, f"actions/game_{game_id}")
    return add_names(actions)


class TestWindowActionE2E:
    def test_scores_shape_and_dtype(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.scores(wc2018_actions, nr_actions=10, window="action")
        assert len(result) == len(wc2018_actions)
        assert result["scores"].dtype == bool

    def test_concedes_shape_and_dtype(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.concedes(wc2018_actions, nr_actions=10, window="action")
        assert len(result) == len(wc2018_actions)
        assert result["concedes"].dtype == bool


class TestWindowPossessionE2E:
    def test_scores_runs(self, wc2018_actions: pd.DataFrame) -> None:
        actions = add_possessions(wc2018_actions)
        result = lab.scores(actions, window="possession")
        assert len(result) == len(actions)
        # WC2018 matches have goals -> at least one True
        assert result["scores"].sum() > 0

    def test_concedes_runs(self, wc2018_actions: pd.DataFrame) -> None:
        actions = add_possessions(wc2018_actions)
        result = lab.concedes(actions, window="possession")
        assert len(result) == len(actions)
        # Concessions within a possession chain require the opponent to score
        # while the current team's actions are in the same chain — this can be
        # zero in a single match (goals often end possession chains)
        assert result["concedes"].dtype == bool


class TestWindowTimeE2E:
    def test_scores_runs(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.scores(wc2018_actions, window="time", window_seconds=15.0)
        assert len(result) == len(wc2018_actions)
        assert result["scores"].sum() > 0

    def test_concedes_runs(self, wc2018_actions: pd.DataFrame) -> None:
        result = lab.concedes(wc2018_actions, window="time", window_seconds=15.0)
        assert len(result) == len(wc2018_actions)
        assert result["concedes"].sum() > 0

    def test_wider_window_more_positives(self, wc2018_actions: pd.DataFrame) -> None:
        narrow = lab.scores(wc2018_actions, window="time", window_seconds=5.0)
        wide = lab.scores(wc2018_actions, window="time", window_seconds=30.0)
        # Wider window should find >= as many scoring situations
        assert wide["scores"].sum() >= narrow["scores"].sum()
