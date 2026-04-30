"""Specialty feature transformers — silly-kicks's own additions beyond the
upstream socceraction feature set.

Two transformers: ``cross_zone`` (categorical zone classification for crosses
into the box), ``assist_type`` (categorical type-of-assist preceding a shot).
"""

import pandas as pd  # type: ignore

import silly_kicks.spadl.config as spadlcfg

from .core import Actions, Features, GameStates, simple

__all__ = ["assist_type", "cross_zone"]


@simple
def cross_zone(actions: Actions) -> Features:
    """Classify the origin zone of cross actions (Gelade 2017).

    Zones:
    - Zone 1 (deep wide): x < 88, wide positions
    - Zone 2 (wide channel): x >= 88, y < 14 or y > 54
    - Zone 3 (byline): x >= 100, y 14-54
    - Zone 4 (half-space/cutback): x >= 88, y 14-54, not byline

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        One-hot encoding of the cross zone (4 columns). Zero for non-cross actions.

    Examples
    --------
    Extract cross-zone categorical features per gamestate slot::

        from silly_kicks.vaep.features import cross_zone

        feats = cross_zone(states)
    """
    is_cross = actions["type_id"].isin(
        [
            spadlcfg.actiontype_id["cross"],
            spadlcfg.actiontype_id["corner_crossed"],
            spadlcfg.actiontype_id["freekick_crossed"],
        ]
    )
    x = actions["start_x"]
    y = actions["start_y"]
    wide = (y < 14) | (y > 54)

    zone1 = is_cross & (x < 88) & wide
    zone2 = is_cross & (x >= 88) & wide
    zone3 = is_cross & (x >= 100) & ~wide
    zone4 = is_cross & (x >= 88) & (x < 100) & ~wide

    return pd.DataFrame(
        {
            "cross_zone_deep_wide": zone1,
            "cross_zone_wide_channel": zone2,
            "cross_zone_byline": zone3,
            "cross_zone_halfspace": zone4,
        },
        index=actions.index,
    )


def assist_type(gamestates: GameStates) -> Features:
    """Classify the assist type for shot actions based on the preceding action.

    Categories based on Carpenter's conversion rate research:
    - through_ball: progressive pass with end_x > 88 and forward movement > 20
    - cutback: cross/pass from byline area (start_x > 99) going backward
    - cross: any cross action
    - set_piece: corner, freekick
    - progressive_pass: forward pass with end_x - start_x > 10
    - other: all other preceding actions

    Only non-zero when the current action (a0) is a shot-like action.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        One-hot encoding of assist type (6 columns). Zero for non-shot a0 actions.

    Examples
    --------
    Extract assist-type categorical features per gamestate slot::

        from silly_kicks.vaep.features import assist_type

        feats = assist_type(states)
    """
    a0 = gamestates[0]
    is_shot = a0["type_id"].isin(
        [
            spadlcfg.actiontype_id["shot"],
            spadlcfg.actiontype_id["shot_penalty"],
            spadlcfg.actiontype_id["shot_freekick"],
        ]
    )

    if len(gamestates) < 2:
        return pd.DataFrame(
            {
                "assist_through_ball": False,
                "assist_cutback": False,
                "assist_cross": False,
                "assist_set_piece": False,
                "assist_progressive_pass": False,
                "assist_other": False,
            },
            index=a0.index,
        )

    a1 = gamestates[1]
    a1_type = a1["type_id"]
    a1_start_x = a1["start_x"]
    a1_end_x = a1["end_x"]

    cross_types = {spadlcfg.actiontype_id[t] for t in ["cross", "corner_crossed", "freekick_crossed"]}
    set_piece_types = {
        spadlcfg.actiontype_id[t]
        for t in [
            "corner_crossed",
            "corner_short",
            "freekick_crossed",
            "freekick_short",
        ]
    }
    pass_types = {spadlcfg.actiontype_id[t] for t in ["pass", "cross"]}

    forward_move = a1_end_x - a1_start_x
    is_cross_a1 = a1_type.isin(cross_types)
    is_set_piece = a1_type.isin(set_piece_types)
    is_pass_a1 = a1_type.isin(pass_types)
    is_cutback = is_shot & ((a1_start_x > 99) & (forward_move < -5))
    is_through = is_shot & is_pass_a1 & (a1_end_x > 88) & (forward_move > 20) & ~is_cutback
    is_progressive = is_shot & is_pass_a1 & (forward_move > 10) & ~is_through & ~is_cutback

    return pd.DataFrame(
        {
            "assist_through_ball": is_shot & is_through,
            "assist_cutback": is_shot & is_cutback,
            "assist_cross": is_shot & is_cross_a1 & ~is_cutback,
            "assist_set_piece": is_shot & is_set_piece & ~is_cutback,
            "assist_progressive_pass": is_shot & is_progressive,
            "assist_other": is_shot & ~(is_through | is_cutback | is_cross_a1 | is_set_piece | is_progressive),
        },
        index=a0.index,
    )
