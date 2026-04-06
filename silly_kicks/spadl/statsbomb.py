"""StatsBomb event stream data to SPADL converter."""

import warnings
from collections import Counter
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances, _fix_direction_of_play
from .schema import ConversionReport
from .utils import _finalize_output, _validate_input_columns

_SB_FIELD_LENGTH: int = 120  # StatsBomb internal grid length
_SB_FIELD_WIDTH: int = 80    # StatsBomb internal grid width

EXPECTED_INPUT_COLUMNS: set[str] = {
    "game_id", "event_id", "period_id", "timestamp",
    "team_id", "player_id", "type_name", "location", "extra",
}

_MAPPED_EVENT_TYPES: frozenset[str] = frozenset({
    "Pass", "Dribble", "Carry", "Foul Committed", "Duel",
    "Interception", "Shot", "Own Goal Against", "Goal Keeper",
    "Clearance", "Miscontrol",
})

_EXCLUDED_EVENT_TYPES: frozenset[str] = frozenset({
    "Pressure", "Ball Receipt*", "Block", "50/50",
    "Substitution", "Starting XI", "Tactical Shift",
    "Referee Ball-Drop", "Half Start", "Half End",
    "Injury Stoppage", "Player On", "Player Off", "Error",
    "Shield", "Camera On", "Camera off",
    "Bad Behaviour", "Ball Recovery",
})


def _flatten_extra(events: pd.DataFrame) -> pd.DataFrame:
    """Extract nested extra dict fields into flat columns for vectorized dispatch."""
    extra = events["extra"]

    def _deep_get(series: pd.Series, *keys: str) -> pd.Series:
        """Chain .str.get() calls for nested dict access."""
        result = series
        for key in keys:
            result = result.str.get(key)
        return result

    events["_pass_type"] = _deep_get(extra, "pass", "type", "name")
    events["_pass_height"] = _deep_get(extra, "pass", "height", "name")
    events["_pass_cross"] = _deep_get(extra, "pass", "cross")
    events["_pass_outcome"] = _deep_get(extra, "pass", "outcome", "name")
    events["_pass_body_part"] = _deep_get(extra, "pass", "body_part", "name")
    events["_pass_end_location"] = _deep_get(extra, "pass", "end_location")
    events["_shot_type"] = _deep_get(extra, "shot", "type", "name")
    events["_shot_outcome"] = _deep_get(extra, "shot", "outcome", "name")
    events["_shot_body_part"] = _deep_get(extra, "shot", "body_part", "name")
    events["_shot_end_location"] = _deep_get(extra, "shot", "end_location")
    events["_dribble_outcome"] = _deep_get(extra, "dribble", "outcome", "name")
    events["_gk_type"] = _deep_get(extra, "goalkeeper", "type", "name")
    events["_gk_outcome"] = _deep_get(extra, "goalkeeper", "outcome", "name")
    events["_gk_body_part"] = _deep_get(extra, "goalkeeper", "body_part", "name")
    events["_foul_card"] = _deep_get(extra, "foul_committed", "card", "name")
    events["_duel_type"] = _deep_get(extra, "duel", "type", "name")
    events["_duel_outcome"] = _deep_get(extra, "duel", "outcome", "name")
    events["_interception_outcome"] = _deep_get(extra, "interception", "outcome", "name")
    events["_clearance_body_part"] = _deep_get(extra, "clearance", "body_part", "name")
    events["_carry_end_location"] = _deep_get(extra, "carry", "end_location")
    return events


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    xy_fidelity_version: Optional[int] = None,
    shot_fidelity_version: Optional[int] = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """
    Convert StatsBomb events to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame containing StatsBomb events from a single game.
    home_team_id : int
        ID of the home team in the corresponding game.
    xy_fidelity_version : int, optional
        Whether low or high fidelity coordinates are used in the event data.
        If not specified, the fidelity version is inferred from the data.
    shot_fidelity_version : int, optional
        Whether low or high fidelity coordinates are used in the event data
        for shots. If not specified, the fidelity version is inferred from the
        data.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding SPADL actions.

    Raises
    ------
    ValueError
        If ``xy_fidelity_version`` or ``shot_fidelity_version`` is not 1 or 2.

    """
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="StatsBomb")
    _event_type_counts = Counter(events["type_name"])

    actions = pd.DataFrame()

    # Determine xy_fidelity_version and shot_fidelity_version
    infered_xy_fidelity_version, infered_shot_fidelity_version = _infer_xy_fidelity_versions(
        events
    )
    if xy_fidelity_version is None:
        xy_fidelity_version = infered_xy_fidelity_version
        warnings.warn(
            f"Inferred xy_fidelity_version={infered_xy_fidelity_version}."
            + " If this is incorrect, please specify the correct version"
            + " using the xy_fidelity_version argument"
        )
    else:
        if xy_fidelity_version not in (1, 2):
            raise ValueError("xy_fidelity_version must be 1 or 2")
    if shot_fidelity_version is None:
        if xy_fidelity_version == 2:
            shot_fidelity_version = 2
        else:
            shot_fidelity_version = infered_shot_fidelity_version
            warnings.warn(
                f"Inferred shot_fidelity_version={infered_shot_fidelity_version}."
                + " If this is incorrect, please specify the correct version"
                + " using the shot_fidelity_version argument"
            )
    else:
        if shot_fidelity_version not in (1, 2):
            raise ValueError("shot_fidelity_version must be 1 or 2")

    events = events.copy()
    events = _insert_interception_passes(events)
    events["extra"] = events["extra"].fillna({})
    events = _flatten_extra(events)

    actions["game_id"] = events.game_id
    actions["original_event_id"] = events.event_id.astype(str)
    actions["period_id"] = events.period_id
    actions["time_seconds"] = pd.to_timedelta(events.timestamp).dt.total_seconds()
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    # split (end)location column into x and y columns
    end_location = events["_pass_end_location"].fillna(
        events["_shot_end_location"]
    ).fillna(
        events["_carry_end_location"]
    ).fillna(
        events["location"]
    )
    # convert StatsBomb coordinates to spadl coordinates
    actions.loc[events.type_name == "Shot", ["start_x", "start_y"]] = _convert_locations(
        events.loc[events.type_name == "Shot", "location"],
        shot_fidelity_version,
    )
    actions.loc[events.type_name != "Shot", ["start_x", "start_y"]] = _convert_locations(
        events.loc[events.type_name != "Shot", "location"],
        shot_fidelity_version,
    )
    actions.loc[events.type_name == "Shot", ["end_x", "end_y"]] = _convert_locations(
        end_location.loc[events.type_name == "Shot"],
        shot_fidelity_version,
    )
    actions.loc[events.type_name != "Shot", ["end_x", "end_y"]] = _convert_locations(
        end_location.loc[events.type_name != "Shot"],
        shot_fidelity_version,
    )

    actions["type_id"] = _vectorized_type_id(events)
    actions["result_id"] = _vectorized_result_id(events)
    actions["bodypart_id"] = _vectorized_bodypart_id(events)

    actions = (
        actions[actions.type_id != spadlconfig.actiontype_id["non_action"]]
        .sort_values(["game_id", "period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )
    actions = _fix_direction_of_play(actions, home_team_id)
    actions = _fix_clearances(actions)

    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions)

    actions = _finalize_output(actions)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for etype, count in _event_type_counts.items():
        if etype in _MAPPED_EVENT_TYPES:
            mapped_counts[etype] = count
        elif etype in _EXCLUDED_EVENT_TYPES:
            excluded_counts[etype] = count
        else:
            unrecognized_counts[etype] = count
    if unrecognized_counts:
        warnings.warn(
            f"StatsBomb: {sum(unrecognized_counts.values())} unrecognized event types "
            f"dropped: {dict(unrecognized_counts)}"
        )
    report = ConversionReport(
        provider="StatsBomb",
        total_events=sum(_event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report


def _insert_interception_passes(df_events: pd.DataFrame) -> pd.DataFrame:
    """Insert interception actions before passes.

    This function converts passes that are also interceptions (type 64) in the
    StatsBomb event data into two separate events, first an interception and
    then a pass.

    Parameters
    ----------
    df_events : pd.DataFrame
        StatsBomb event dataframe

    Returns
    -------
    pd.DataFrame
        StatsBomb event dataframe in which passes that were also denoted as
        interceptions in the StatsBomb notation are transformed into two events.
    """
    extra = df_events["extra"].fillna({})
    pass_type = extra.str.get("pass").str.get("type").str.get("name")
    mask = pass_type == "Interception"

    df_events_interceptions = df_events[mask].copy()

    if not df_events_interceptions.empty:
        df_events_interceptions["type_name"] = "Interception"
        df_events_interceptions["extra"] = [
            {"interception": {"outcome": {"id": 16, "name": "Success In Play"}}}
        ] * len(df_events_interceptions)

        df_events = pd.concat([df_events_interceptions, df_events], ignore_index=True)
        df_events = df_events.sort_values(["timestamp"], kind="mergesort")
        df_events = df_events.reset_index(drop=True)

    return df_events


def _infer_xy_fidelity_versions(events: pd.DataFrame) -> tuple[int, int]:
    """Find out if x and y are integers disguised as floats."""
    mask_shot = events.type_name == "Shot"
    mask_other = events.type_name != "Shot"
    valid_locs = events.location.dropna()
    if len(valid_locs) == 0:
        return 1, 1
    locations = pd.DataFrame(valid_locs.tolist(), index=valid_locs.index)
    mask_valid_location = locations.notna().any(axis=1)
    shot_mask = mask_valid_location & mask_shot.reindex(locations.index, fill_value=False)
    other_mask = mask_valid_location & mask_other.reindex(locations.index, fill_value=False)
    high_fidelity_shots = (locations.loc[shot_mask] % 1 != 0).any(axis=None)
    high_fidelity_other = (locations.loc[other_mask] % 1 != 0).any(axis=None)
    xy_fidelity_version = 2 if high_fidelity_other else 1
    shot_fidelity_version = 2 if high_fidelity_shots else xy_fidelity_version
    return shot_fidelity_version, xy_fidelity_version


def _convert_locations(locations: pd.Series, fidelity_version: int) -> npt.NDArray[np.float64]:
    """Convert StatsBomb locations to spadl coordinates.

    StatsBomb coordinates are cell-based, using a 120x80 grid, so 1,1 is the
    top-left square 'yard' of the field (in landscape), even though 0,0 is the
    true coordinate of the corner flag.

    Some matches have metadata like "xy_fidelity_version" : "2", which means
    the grid has higher granularity. In this case 0.1,0.1 is the top left
    cell.
    """
    # [1, 120] x [1, 80]
    # +-----+------+
    # | 1,1 | 2, 1 |
    # +-----+------+
    # | 1,2 | 2,2  |
    # +-----+------+
    n = len(locations)
    if n == 0:
        return np.empty((0, 2), dtype=float)
    cell_side = 0.1 if fidelity_version == 2 else 1.0
    crc = cell_side / 2
    loc_list = locations.tolist()
    xy_raw = np.array(
        [loc[:2] if isinstance(loc, list) and len(loc) >= 2 else [np.nan, np.nan] for loc in loc_list],
        dtype=float,
    )
    is_three = np.array([isinstance(loc, list) and len(loc) == 3 for loc in loc_list])
    y_offset = np.where(is_three, 0.05, crc)
    coordinates = np.empty((n, 2), dtype=float)
    coordinates[:, 0] = (xy_raw[:, 0] - crc) / _SB_FIELD_LENGTH * spadlconfig.field_length
    coordinates[:, 1] = spadlconfig.field_width - (xy_raw[:, 1] - y_offset) / _SB_FIELD_WIDTH * spadlconfig.field_width
    coordinates[:, 0] = np.clip(coordinates[:, 0], 0, spadlconfig.field_length)
    coordinates[:, 1] = np.clip(coordinates[:, 1], 0, spadlconfig.field_width)
    return coordinates


def _vectorized_type_id(events: pd.DataFrame) -> pd.Series:
    """Compute SPADL type_id for all events using vectorized np.select."""
    t = events["type_name"]
    _at = spadlconfig.actiontype_id
    non_action = _at["non_action"]

    # Pre-fetch flattened columns
    pass_type = events["_pass_type"]
    pass_height = events["_pass_height"]
    pass_cross = events["_pass_cross"]
    pass_outcome = events["_pass_outcome"]
    duel_type = events["_duel_type"]
    shot_type = events["_shot_type"]
    gk_type = events["_gk_type"]

    is_pass = t == "Pass"
    high_or_cross = (pass_height == "High Pass") | (pass_cross == True)  # noqa: E712

    # Pass sub-types — order matters (first match wins in np.select)
    conditions = [
        # Pass with outcome "Injury Clearance" or "Unknown" → non_action
        is_pass & pass_outcome.isin(["Injury Clearance", "Unknown"]),
        # Free Kick pass: crossed or short
        is_pass & (pass_type == "Free Kick") & high_or_cross,
        is_pass & (pass_type == "Free Kick") & ~high_or_cross,
        # Corner: crossed or short
        is_pass & (pass_type == "Corner") & high_or_cross,
        is_pass & (pass_type == "Corner") & ~high_or_cross,
        # Goal Kick
        is_pass & (pass_type == "Goal Kick"),
        # Throw-in
        is_pass & (pass_type == "Throw-in"),
        # Cross (not a set piece)
        is_pass & (pass_cross == True) & ~pass_type.isin(["Free Kick", "Corner", "Goal Kick", "Throw-in"]),  # noqa: E712
        # Regular pass (fallthrough for all other passes)
        is_pass,
        # Dribble (StatsBomb) → take_on
        t == "Dribble",
        # Carry → dribble
        t == "Carry",
        # Foul Committed
        t == "Foul Committed",
        # Duel — Tackle
        (t == "Duel") & (duel_type == "Tackle"),
        # Duel — non-tackle → non_action
        (t == "Duel") & (duel_type != "Tackle"),
        # Interception
        t == "Interception",
        # Shot sub-types
        (t == "Shot") & (shot_type == "Free Kick"),
        (t == "Shot") & (shot_type == "Penalty"),
        t == "Shot",
        # Own Goal Against
        t == "Own Goal Against",
        # Goalkeeper sub-types
        (t == "Goal Keeper") & (gk_type == "Shot Saved"),
        (t == "Goal Keeper") & gk_type.isin(["Collected", "Keeper Sweeper"]),
        (t == "Goal Keeper") & (gk_type == "Punch"),
        t == "Goal Keeper",  # fallthrough → non_action
        # Clearance
        t == "Clearance",
        # Miscontrol
        t == "Miscontrol",
    ]

    choices = [
        non_action,                    # Pass Injury Clearance/Unknown
        _at["freekick_crossed"],
        _at["freekick_short"],
        _at["corner_crossed"],
        _at["corner_short"],
        _at["goalkick"],
        _at["throw_in"],
        _at["cross"],
        _at["pass"],
        _at["take_on"],
        _at["dribble"],
        _at["foul"],
        _at["tackle"],
        non_action,                    # Duel non-tackle
        _at["interception"],
        _at["shot_freekick"],
        _at["shot_penalty"],
        _at["shot"],
        _at["bad_touch"],              # Own Goal Against
        _at["keeper_save"],
        _at["keeper_claim"],
        _at["keeper_punch"],
        non_action,                    # Goal Keeper fallthrough
        _at["clearance"],
        _at["bad_touch"],              # Miscontrol
    ]

    return pd.Series(np.select(conditions, choices, default=non_action), index=events.index)


def _vectorized_result_id(events: pd.DataFrame) -> pd.Series:
    """Compute SPADL result_id for all events using vectorized np.select."""
    t = events["type_name"]
    _r = spadlconfig.result_id

    pass_outcome = events["_pass_outcome"]
    dribble_outcome = events["_dribble_outcome"]
    duel_type = events["_duel_type"]
    duel_outcome = events["_duel_outcome"]
    interception_outcome = events["_interception_outcome"]
    shot_outcome = events["_shot_outcome"]
    foul_card = events["_foul_card"].fillna("")
    gk_outcome = events["_gk_outcome"].fillna("x")

    is_pass = t == "Pass"

    conditions = [
        # Pass results
        is_pass & pass_outcome.isin(["Incomplete", "Out"]),
        is_pass & (pass_outcome == "Pass Offside"),
        is_pass & pass_outcome.isin(["Injury Clearance", "Unknown"]),
        is_pass,
        # Dribble
        (t == "Dribble") & (dribble_outcome == "Incomplete"),
        t == "Dribble",
        # Carry
        t == "Carry",
        # Foul Committed
        (t == "Foul Committed") & foul_card.str.contains("Yellow", na=False),
        (t == "Foul Committed") & foul_card.str.contains("Red", na=False),
        t == "Foul Committed",
        # Duel — Tackle
        (t == "Duel") & (duel_type == "Tackle") & duel_outcome.isin(["Lost In Play", "Lost Out"]),
        (t == "Duel") & (duel_type == "Tackle"),
        # Duel — non-tackle → non_action result (success)
        t == "Duel",
        # Interception
        (t == "Interception") & interception_outcome.isin(["Lost In Play", "Lost Out"]),
        t == "Interception",
        # Shot
        (t == "Shot") & (shot_outcome == "Goal"),
        t == "Shot",
        # Own Goal Against
        t == "Own Goal Against",
        # Goal Keeper
        (t == "Goal Keeper") & gk_outcome.isin(["In Play Danger", "No Touch"]),
        t == "Goal Keeper",
        # Clearance
        t == "Clearance",
        # Miscontrol
        t == "Miscontrol",
    ]

    choices = [
        _r["fail"],                    # Pass Incomplete/Out
        _r["offside"],                 # Pass Offside
        _r["success"],                 # Pass Injury Clearance/Unknown (non_action)
        _r["success"],                 # Pass default
        _r["fail"],                    # Dribble Incomplete
        _r["success"],                 # Dribble default
        _r["success"],                 # Carry
        _r["yellow_card"],             # Foul Yellow
        _r["red_card"],                # Foul Red
        _r["fail"],                    # Foul default
        _r["fail"],                    # Tackle Lost
        _r["success"],                 # Tackle default
        _r["success"],                 # Duel non-tackle → non_action success
        _r["fail"],                    # Interception Lost
        _r["success"],                 # Interception default
        _r["success"],                 # Shot Goal
        _r["fail"],                    # Shot default (not Goal)
        _r["owngoal"],                 # Own Goal Against
        _r["fail"],                    # GK In Play Danger / No Touch
        _r["success"],                 # GK default
        _r["success"],                 # Clearance
        _r["fail"],                    # Miscontrol
    ]

    return pd.Series(np.select(conditions, choices, default=_r["success"]), index=events.index)


def _vectorized_bodypart_id(events: pd.DataFrame) -> pd.Series:
    """Compute SPADL bodypart_id for all events using vectorized np.select."""
    t = events["type_name"]
    _b = spadlconfig.bodypart_id

    pass_bp = events["_pass_body_part"].fillna("")
    shot_bp = events["_shot_body_part"].fillna("")
    gk_bp = events["_gk_body_part"].fillna("")
    clearance_bp = events["_clearance_body_part"].fillna("")
    pass_type = events["_pass_type"]

    # Helper: body part resolution for pass
    # Default is "foot", unless type is "Throw-in" → "other"
    # Then overridden if _pass_body_part is not None (not "")
    is_pass = t == "Pass"
    pass_bp_notna = events["_pass_body_part"].notna()

    # Helper: body part resolution for shot (default "foot", override if not None)
    is_shot = t == "Shot"
    shot_bp_notna = events["_shot_body_part"].notna()

    # Helper: body part resolution for goalkeeper (default "other", override if not None)
    is_gk = t == "Goal Keeper"
    gk_bp_notna = events["_gk_body_part"].notna()

    # Helper: body part resolution for clearance (default "foot", override if not None)
    is_clearance = t == "Clearance"
    clearance_bp_notna = events["_clearance_body_part"].notna()

    conditions = [
        # Pass with body part specified
        is_pass & pass_bp_notna & pass_bp.str.contains("Head", na=False),
        is_pass & pass_bp_notna & (pass_bp == "Left Foot"),
        is_pass & pass_bp_notna & (pass_bp == "Right Foot"),
        is_pass & pass_bp_notna & (pass_bp.str.contains("Foot", na=False) | (pass_bp == "Drop Kick")),
        is_pass & pass_bp_notna,  # other body part
        # Pass without body part: Throw-in → other, else foot
        is_pass & (pass_type == "Throw-in"),
        is_pass,
        # Shot with body part specified
        is_shot & shot_bp_notna & shot_bp.str.contains("Head", na=False),
        is_shot & shot_bp_notna & (shot_bp == "Left Foot"),
        is_shot & shot_bp_notna & (shot_bp == "Right Foot"),
        is_shot & shot_bp_notna & shot_bp.str.contains("Foot", na=False),
        is_shot & shot_bp_notna,  # other
        is_shot,  # default foot
        # Goalkeeper with body part specified
        is_gk & gk_bp_notna & gk_bp.str.contains("Head", na=False),
        is_gk & gk_bp_notna & (gk_bp == "Left Foot"),
        is_gk & gk_bp_notna & (gk_bp == "Right Foot"),
        is_gk & gk_bp_notna & (gk_bp.str.contains("Foot", na=False) | (gk_bp == "Drop Kick")),
        is_gk & gk_bp_notna,  # other
        is_gk,  # default other
        # Clearance with body part specified
        is_clearance & clearance_bp_notna & clearance_bp.str.contains("Head", na=False),
        is_clearance & clearance_bp_notna & (clearance_bp == "Left Foot"),
        is_clearance & clearance_bp_notna & (clearance_bp == "Right Foot"),
        is_clearance & clearance_bp_notna & clearance_bp.str.contains("Foot", na=False),
        is_clearance & clearance_bp_notna,  # other
        is_clearance,  # default foot
    ]

    choices = [
        _b["head"],
        _b["foot_left"],
        _b["foot_right"],
        _b["foot"],
        _b["other"],
        _b["other"],       # Throw-in default
        _b["foot"],        # Pass default
        _b["head"],
        _b["foot_left"],
        _b["foot_right"],
        _b["foot"],
        _b["other"],
        _b["foot"],        # Shot default
        _b["head"],
        _b["foot_left"],
        _b["foot_right"],
        _b["foot"],
        _b["other"],
        _b["other"],       # GK default
        _b["head"],
        _b["foot_left"],
        _b["foot_right"],
        _b["foot"],
        _b["other"],
        _b["foot"],        # Clearance default
    ]

    # Default: foot for all other event types (Dribble, Carry, Foul, Duel,
    # Interception, Own Goal Against, Miscontrol, and non_action)
    return pd.Series(np.select(conditions, choices, default=_b["foot"]), index=events.index)
