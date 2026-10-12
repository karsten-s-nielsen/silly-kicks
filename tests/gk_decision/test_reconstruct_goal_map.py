"""ReconstructedOptionSet(goal_map=): inject a whole-unit GoalMap for the match-LTR reflection.

In the ``match_ltr`` convention ``option_rows`` resolves the defended-goal end to decide whether each
frame is reflected into action-LTR. Under the lakehouse's windowed Stage-2 scoring that resolution
would run per window; ``goal_map=`` lets the caller inject the whole-unit map so the reflection is
stable across windows. ``goal_map=None`` recomputes internally -> byte-identical to the pre-parameter
behaviour. (The ``per_action_ltr`` branch never touches the goal map, so it is unaffected either way.)
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pandas as pd

from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import GoalMap, resolve_defended_goals


class _FakeXPass:
    """Deterministic distance-decay completion so the reflected vs unreflected rows differ exactly."""

    def predict_completion(self, ox, oy, tx, ty):
        ox, oy, tx, ty = (np.asarray(a, dtype=float) for a in (ox, oy, tx, ty))
        return np.clip(1.0 - np.hypot(tx - ox, ty - oy) / 120.0, 0.0, 1.0)


def _match_ltr_decision():
    """One goalkick, match-LTR: team 7 keeper attacks x=105, team 8 keeper defends x=105.

    Both teams carry a keeper so ``resolve_defended_goals`` resolves both ends cleanly (team 7 -> 0,
    team 8 -> 105) and ``attacked_goal(team 7)`` is 105 -> the frame is NOT reflected by default.
    """
    actions = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                action_id=1,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=40.0,
                end_y=34.0,
            )
        ]
    )
    frames = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=99,
                team_id=7,
                x=10.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=50,
                team_id=7,
                x=40.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=88,
                team_id=8,
                x=95.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=77,
                team_id=8,
                x=50.0,
                y=30.0,
                is_ball=False,
                is_goalkeeper=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=1,
                team_id=None,
                x=10.0,
                y=34.0,
                is_ball=True,
                is_goalkeeper=False,
            ),
        ]
    )
    links = pd.DataFrame({"action_id": [1], "frame_id": [1]})
    return actions, frames, links


def _optionset(actions, frames, links, **kwargs) -> ReconstructedOptionSet:
    return ReconstructedOptionSet(
        actions,
        frames,
        xpass=_FakeXPass(),
        params=GkDecisionParams(),
        keeper_ids=[99],
        frame_convention="match_ltr",
        links=links,
        **kwargs,
    )


def _invert(gm: GoalMap) -> GoalMap:
    fl = float(spadlconfig.field_length)

    def _swap(m) -> MappingProxyType:
        return MappingProxyType({k: fl - v for k, v in m.items()})

    return GoalMap(_swap(gm.resolved), _swap(gm.guessed), gm.unresolved)


def test_goal_map_none_is_byte_identical_to_the_whole_match_recompute():
    actions, frames, links = _match_ltr_decision()
    whole = resolve_defended_goals(frames)

    default = _optionset(actions, frames, links).option_rows()
    none = _optionset(actions, frames, links, goal_map=None).option_rows()
    inject = _optionset(actions, frames, links, goal_map=whole).option_rows()

    assert len(default) >= 1 and int(default["is_chosen"].sum()) == 1, "non-vacuity: one chosen option"
    assert none.equals(default)
    assert inject.equals(default)


def test_goal_map_is_load_bearing_match_ltr():
    actions, frames, links = _match_ltr_decision()
    inverted = _invert(resolve_defended_goals(frames))

    default = _optionset(actions, frames, links).option_rows()
    flipped = _optionset(actions, frames, links, goal_map=inverted).option_rows()

    assert len(default) >= 1 and int(default["is_chosen"].sum()) == 1
    # Inverting the defended ends flips the match-LTR reflection -> the keeper moves -> completion
    # changes; a silently-ignored parameter would leave the rows identical.
    assert not flipped.equals(default)
