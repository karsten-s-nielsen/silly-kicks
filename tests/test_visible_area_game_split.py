"""Every ``visible_area`` consumer keeps the game-split property with game-specific polygons (spec W9.6).

Owner ruling 2026-11-12: the five consumers read polygons through ONE join, ``silly_kicks._visible_area``. Two games
share ``action_id`` values but carry different polygons. With a ``game_id`` column in ``visible_area``, the two-game
call equals the one-game calls. Without one, the call refuses: an ambiguous join is never guessed.

``compute_territorial_defense`` is in the refusal test only: on this fixture no candidate reaches its polygon check
(every one is ``no_actor``), so its split property is proven on the engine fixture that does reach it
(``tests/territorial_defense/test_engine.py``).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from scripts._sb_battery import audit_xt
from tests.invariants._link_population import COMPARATORS, comparator_for, split_by_game
from tests.test_multigame_linking import make_two_game_inputs

LEFT_HALF = np.array([[0.0, 0.0], [52.5, 0.0], [52.5, 68.0], [0.0, 68.0]])
WHOLE_PITCH = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])
RIGHT_THIRD = np.array([[70.0, 0.0], [105.0, 0.0], [105.0, 68.0], [70.0, 68.0]])


def _visible_area(actions: pd.DataFrame, polygon: np.ndarray, *, with_game: bool = True) -> pd.DataFrame:
    table = pd.DataFrame({"action_id": actions["action_id"].to_numpy(), "polygon": [polygon] * len(actions)})
    if with_game:
        table.insert(0, "game_id", actions["game_id"].to_numpy())
    return table


def _with_xg(actions: pd.DataFrame) -> pd.DataFrame:
    out = actions.copy()
    out["audit_xg"] = 0.12
    return out


def _calls():
    from silly_kicks.restdefense import compute_rest_defense
    from silly_kicks.territorial_defense._compute import compute_territorial_defense
    from silly_kicks.tracking import add_action_context, add_defensive_credit, add_visible_area_coverage

    xt = audit_xt()
    return {
        "silly_kicks.tracking._visibility:add_visible_area_coverage": lambda a, f, va: add_visible_area_coverage(
            a, visible_area=va
        ),
        "silly_kicks.tracking.features:add_action_context": lambda a, f, va: add_action_context(a, f, visible_area=va),
        "silly_kicks.tracking.features:add_defensive_credit": lambda a, f, va: add_defensive_credit(
            _with_xg(a), f, xg_column="audit_xg", xt=xt, visible_area=va
        ),
        "silly_kicks.territorial_defense._compute:compute_territorial_defense": lambda a, f, va: (
            compute_territorial_defense(a, f, xt=xt, visible_area=va)
        ),
        "silly_kicks.restdefense._compute:compute_rest_defense": lambda a, f, va: compute_rest_defense(
            a, f, xt=xt, visible_area=va
        ),
    }


_MEMBERS = sorted(_calls())
_TERRITORIAL = "silly_kicks.territorial_defense._compute:compute_territorial_defense"


def _table(result) -> pd.DataFrame:
    return result[0] if isinstance(result, tuple) else result


@pytest.mark.parametrize("member", [m for m in _MEMBERS if m != _TERRITORIAL])
def test_game_specific_polygons_keep_the_split_property(member):
    call = _calls()[member]
    actions, frames = make_two_game_inputs()
    (a, fa), (b, fb) = split_by_game(actions, frames)
    va_a, va_b = _visible_area(a, LEFT_HALF), _visible_area(b, WHOLE_PITCH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = [call(a, fa, va_a), call(b, fb, va_b)]
        whole = call(actions, frames, pd.concat([va_a, va_b], ignore_index=True))
        swapped = call(a, fa, _visible_area(a, RIGHT_THIRD))
    assert not _table(swapped).equals(_table(parts[0])), "non-vacuity: game a's polygon must change its output"
    COMPARATORS[comparator_for(member)].assert_split_equal(whole, parts, actions)


@pytest.mark.parametrize("member", _MEMBERS)
def test_two_games_without_a_game_id_column_refuse(member):
    call = _calls()[member]
    actions, frames = make_two_game_inputs()
    with warnings.catch_warnings(), pytest.raises(ValueError, match="game_id"):
        warnings.simplefilter("ignore")
        call(actions, frames, _visible_area(actions, LEFT_HALF, with_game=False))
