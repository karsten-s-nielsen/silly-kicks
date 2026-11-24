"""The ONE visible-area polygon join (owner ruling 2026-11-12; spec W9.4, ADR-019).

An ``action_id`` names one action only inside a game. A ``visible_area`` table with a ``game_id`` column joins on
``(game_id, action_id)``; without one, the ``action_id`` join is allowed only when it is unambiguous (the W9.5
legacy-links rule). An ambiguous join raises: it is never guessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks._visible_area import polygons_by_action_row

A = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
B = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])


def _two_games() -> pd.DataFrame:
    return pd.DataFrame({"game_id": [1, 1, 2, 2], "action_id": [0, 1, 0, 1]})


def test_a_game_id_column_joins_each_row_to_its_own_games_polygon():
    visible = pd.DataFrame({"game_id": [1, 2, 2], "action_id": [0, 0, 1], "polygon": [A, B, B]})
    polys = polygons_by_action_row(_two_games(), visible)
    assert polys[0] is A and polys[2] is B and polys[3] is B
    assert polys[1] is None, "game 1 action 1 has no polygon; game 2's action 1 must not stand in for it"


def test_the_game_join_is_dtype_invariant():
    actions = _two_games().astype({"game_id": "string", "action_id": "float64"})
    visible = pd.DataFrame({"game_id": [1, 2], "action_id": ["0", "1"], "polygon": [A, B]})
    polys = polygons_by_action_row(actions, visible)
    assert polys[0] is A and polys[3] is B and polys[1] is None and polys[2] is None


def test_without_game_id_one_game_joins_on_action_id():
    actions = pd.DataFrame({"game_id": [7, 7], "action_id": [0, 1]})
    visible = pd.DataFrame({"action_id": [1], "polygon": [B]})
    assert polygons_by_action_row(actions, visible) == [None, B]
    assert polygons_by_action_row(actions.drop(columns="game_id"), visible) == [None, B]


def test_without_game_id_two_games_refuse():
    visible = pd.DataFrame({"action_id": [0, 1], "polygon": [A, B]})
    with pytest.raises(ValueError, match="game_id"):
        polygons_by_action_row(_two_games(), visible)


@pytest.mark.parametrize("with_game", [False, True])
def test_a_repeated_visible_area_key_refuses(with_game):
    visible = pd.DataFrame({"action_id": [0, 0], "polygon": [A, B]})
    if with_game:
        visible.insert(0, "game_id", [1, 1])
    with pytest.raises(ValueError, match="repeat"):
        polygons_by_action_row(_two_games() if with_game else pd.DataFrame({"action_id": [0, 1]}), visible)


def test_a_repeated_key_names_the_frame_that_lacks_game_id():
    """Review B TB-IMPL-82: visible_area has game_id and actions has none. Two games' polygons for one action_id
    send the caller to actions; a duplicate polygon inside one game is a repeated key."""
    actions = pd.DataFrame({"action_id": [0, 1]})
    two_games = pd.DataFrame({"game_id": [1, 2], "action_id": [0, 0], "polygon": [A, B]})
    with pytest.raises(ValueError) as caught:
        polygons_by_action_row(actions, two_games)
    assert str(caught.value) == (
        "visible_area repeats the action_id '0' in two games, and actions has no game_id column to tell them "
        "apart. Add a game_id column to actions, or call once per game."
    )
    one_game = two_games.assign(game_id=[1, 1])
    with pytest.raises(ValueError) as caught:
        polygons_by_action_row(actions, one_game)
    assert str(caught.value) == "visible_area repeats the key '0': one action has more than one polygon."


def test_a_repeated_action_id_without_game_id_refuses_the_action_id_join():
    """With no game_id column, a repeated action_id may be a second game: the join cannot tell, so it refuses."""
    with pytest.raises(ValueError, match="repeat"):
        polygons_by_action_row(pd.DataFrame({"action_id": [0, 0]}), pd.DataFrame({"action_id": [0], "polygon": [A]}))


def test_a_repeated_action_id_in_one_known_game_reads_the_same_polygon():
    """One game_id: a repeated row is the same action, so both rows read its polygon (no ambiguity)."""
    actions = pd.DataFrame({"game_id": [7, 7, 7], "action_id": [0, 0, 1]})
    assert polygons_by_action_row(actions, pd.DataFrame({"action_id": [0], "polygon": [A]})) == [A, A, None]


@pytest.mark.parametrize("visible", [None, pd.DataFrame(columns=["action_id", "polygon"])])
def test_no_visible_area_gives_no_polygon_and_never_raises(visible):
    assert polygons_by_action_row(_two_games(), visible) == [None] * 4
