"""TF-65 C2: xcross --data-dir loader home override.

xcross signs ``score_differential`` from HOME's perspective, so the first-non-ball ``team_id``
heuristic flips the sign on a match whose frames lead with the away team. A tree built from a flat
for_each generation carries no meta.json, so ``--home-teams`` (a ``{game_id: home_team_id}`` map)
must override the heuristic. Falls back to the first team when the map is absent or lacks the game.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.train_xcross_attempt import _iter_matches_from_dir


def _write_game(d: Path, game_id: str = "g7") -> None:
    d.mkdir(parents=True, exist_ok=True)
    # first non-ball team_id is AWAY -> the first-team fallback would pick AWAY
    pd.DataFrame(
        {
            "game_id": [game_id, game_id, game_id],
            "is_ball": [True, False, False],
            "team_id": [pd.NA, "AWAY", "HOME"],
            "source_provider": ["skillcorner", "skillcorner", "skillcorner"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 3.0, 4.0],
        }
    ).to_parquet(d / "frames.parquet")
    pd.DataFrame({"game_id": [game_id], "type_name": ["cross"]}).to_parquet(d / "actions.parquet")


def test_home_map_overrides_first_team(tmp_path):
    _write_game(tmp_path / "g7")
    ((prov, mid, *_, home),) = list(_iter_matches_from_dir(tmp_path, {"g7": "HOME"}))
    assert prov == "skillcorner"
    assert mid == "g7"
    assert home == "HOME"


def test_falls_back_to_first_team_without_map(tmp_path):
    _write_game(tmp_path / "g7")
    ((*_, home),) = list(_iter_matches_from_dir(tmp_path, None))
    assert home == "AWAY"


def test_falls_back_when_game_absent_from_map(tmp_path):
    _write_game(tmp_path / "g7")
    ((*_, home),) = list(_iter_matches_from_dir(tmp_path, {"other": "X"}))
    assert home == "AWAY"
