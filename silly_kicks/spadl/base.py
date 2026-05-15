"""Utility functions for all event stream to SPADL converters.

A converter should implement 'convert_to_actions' to convert the events to the
SPADL format.

"""

import pandas as pd  # type: ignore

from . import config as spadlconfig

# Type IDs for pass-class actions where the ball physically travels to a
# different location.  Used by _derive_end_coordinates to overwrite
# placeholder end_x/end_y with the next action's start position.
_DERIVE_END_TYPE_IDS: frozenset[int] = frozenset(
    {
        spadlconfig.actiontype_id["pass"],  # 0
        spadlconfig.actiontype_id["cross"],  # 1
        spadlconfig.actiontype_id["throw_in"],  # 2
        spadlconfig.actiontype_id["freekick_crossed"],  # 3
        spadlconfig.actiontype_id["freekick_short"],  # 4
        spadlconfig.actiontype_id["corner_crossed"],  # 5
        spadlconfig.actiontype_id["corner_short"],  # 6
        spadlconfig.actiontype_id["clearance"],  # 18
        spadlconfig.actiontype_id["goalkick"],  # 22
    }
)


def _derive_end_coordinates(actions: pd.DataFrame) -> pd.DataFrame:
    """Derive end_x/end_y from next action's start for pass-class types.

    Only overwrites rows where the source data did not provide a separate
    end coordinate (detected by ``end_x == start_x AND end_y == start_y``,
    or ``end_x`` is NaN).
    Period-safe: uses ``groupby("period_id").shift(-1)`` so the last action
    per period keeps its original end coordinates.

    Replaces the former ``_fix_clearances`` with a broader type set, a
    source-data guard, and period-boundary safety.
    """
    if len(actions) == 0:
        return actions
    actions = actions.copy()

    is_pass_class = actions["type_id"].isin(_DERIVE_END_TYPE_IDS)
    placeholder_end = (actions["end_x"] == actions["start_x"]) & (actions["end_y"] == actions["start_y"])
    missing_end = actions["end_x"].isna()
    needs_derivation = is_pass_class & (placeholder_end | missing_end)

    next_start_x = actions.groupby("period_id")["start_x"].shift(-1)
    next_start_y = actions.groupby("period_id")["start_y"].shift(-1)

    mask = needs_derivation & next_start_x.notna()
    actions.loc[mask, "end_x"] = next_start_x[mask].values
    actions.loc[mask, "end_y"] = next_start_y[mask].values
    return actions


min_dribble_length: float = 3.0
max_dribble_length: float = 60.0
max_dribble_duration: float = 10.0


def _add_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = actions.shift(-1)

    same_team = actions.team_id == next_actions.team_id
    # not_clearance = actions.type_id != actiontypes.index("clearance")
    not_offensive_foul = same_team & (next_actions.type_id != spadlconfig.actiontype_id["foul"])
    not_headed_shot = (next_actions.type_id != spadlconfig.actiontype_id["shot"]) & (
        next_actions.bodypart_id != spadlconfig.bodypart_id["head"]
    )

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2
    not_too_far = dx**2 + dy**2 <= max_dribble_length**2

    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < max_dribble_duration
    same_period = actions.period_id == next_actions.period_id

    dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period & not_offensive_foul & not_headed_shot

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles["game_id"] = nex.game_id
    dribbles["period_id"] = nex.period_id
    dribbles["action_id"] = prev.action_id + 0.1
    dribbles["time_seconds"] = (prev.time_seconds + nex.time_seconds) / 2
    if "timestamp" in actions.columns:
        dribbles["timestamp"] = nex.timestamp
    dribbles["team_id"] = nex.team_id
    dribbles["player_id"] = nex.player_id
    dribbles["start_x"] = prev.end_x
    dribbles["start_y"] = prev.end_y
    dribbles["end_x"] = nex.start_x
    dribbles["end_y"] = nex.start_y
    dribbles["bodypart_id"] = spadlconfig.bodypart_id["foot"]
    dribbles["type_id"] = spadlconfig.actiontype_id["dribble"]
    dribbles["result_id"] = spadlconfig.result_id["success"]

    actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)  # type: ignore[reportAssignmentType]
    actions["action_id"] = range(len(actions))
    return actions
