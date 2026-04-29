"""Metrica DataFrame SPADL converter.

Converts already-normalized Metrica event DataFrames (e.g., parsed from
Metrica's open-data CSV / EPTS-JSON formats) to SPADL actions.

Consumers with raw Metrica files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.metrica.load_event(...)``.
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances, _fix_direction_of_play
from .schema import KLOPPY_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns, _validate_preserve_native

EXPECTED_INPUT_COLUMNS: set[str] = {
    "match_id",
    "event_id",
    "type",
    "subtype",
    "period",
    "start_time_s",
    "end_time_s",
    "player",
    "team",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
}

_MAPPED_TYPES: frozenset[str] = frozenset(
    {
        "PASS",
        "SHOT",
        "RECOVERY",
        "CHALLENGE",
        "BALL LOST",
        "FAULT",
        "SET PIECE",
    }
)

_EXCLUDED_TYPES: frozenset[str] = frozenset(
    {
        "BALL OUT",
        "FAULT RECEIVED",
        "CARD",
        "SUBSTITUTION",
    }
)


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert normalized Metrica event DataFrame to SPADL actions."""
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Metrica")
    _validate_preserve_native(events, preserve_native, provider="Metrica", schema=KLOPPY_SPADL_COLUMNS)

    event_type_counts = Counter(events["type"])

    raw_actions = _build_raw_actions(events, preserve_native)

    if len(raw_actions) > 0:
        actions = _fix_clearances(raw_actions)
        actions = _fix_direction_of_play(actions, home_team_id)
        actions["action_id"] = range(len(actions))
        actions = _add_dribbles(actions)

        actions["start_x"] = actions["start_x"].clip(0, spadlconfig.field_length)
        actions["start_y"] = actions["start_y"].clip(0, spadlconfig.field_width)
        actions["end_x"] = actions["end_x"].clip(0, spadlconfig.field_length)
        actions["end_y"] = actions["end_y"].clip(0, spadlconfig.field_width)
    else:
        actions = raw_actions

    extras = list(preserve_native) if preserve_native else None
    actions = _finalize_output(actions, schema=KLOPPY_SPADL_COLUMNS, extra_columns=extras)

    mapped_counts: dict[str, int] = {}
    excluded_counts: dict[str, int] = {}
    unrecognized_counts: dict[str, int] = {}
    for et, count in event_type_counts.items():
        label = str(et)
        if et in _MAPPED_TYPES:
            mapped_counts[label] = count
        elif et in _EXCLUDED_TYPES:
            excluded_counts[label] = count
        else:
            unrecognized_counts[label] = count
    if unrecognized_counts:
        warnings.warn(
            f"Metrica: {sum(unrecognized_counts.values())} unrecognized types dropped: {dict(unrecognized_counts)}",
            stacklevel=2,
        )
    report = ConversionReport(
        provider="Metrica",
        total_events=sum(event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report


def _empty_raw_actions(preserve_native: list[str] | None, events: pd.DataFrame) -> pd.DataFrame:
    cols = {col: pd.Series(dtype=dtype) for col, dtype in KLOPPY_SPADL_COLUMNS.items()}
    if preserve_native:
        for col in preserve_native:
            cols[col] = pd.Series(dtype=events[col].dtype if col in events.columns else "object")
    return pd.DataFrame(cols)


def _apply_card_pairs(actions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Upgrade FAULT-derived foul rows' result_id when paired with a CARD row (≤ 3s, same player)."""
    if "type" not in events.columns:
        return actions
    cards = events[events["type"] == "CARD"]
    if cards.empty:
        return actions

    foul_id = spadlconfig.actiontype_id["foul"]
    foul_actions = actions[actions["type_id"] == foul_id]
    for idx, foul_row in foul_actions.iterrows():
        candidates = cards[
            (cards["period"] == foul_row["period_id"])
            & (cards["player"] == foul_row["player_id"])
            & (cards["start_time_s"] >= foul_row["time_seconds"])
            & (cards["start_time_s"] <= foul_row["time_seconds"] + 3.0)
        ]
        if candidates.empty:
            continue
        sub = str(candidates.iloc[0]["subtype"]).upper()
        if "RED" in sub:
            actions.loc[idx, "result_id"] = spadlconfig.result_id["red_card"]
        elif "YELLOW" in sub:
            actions.loc[idx, "result_id"] = spadlconfig.result_id["yellow_card"]
    return actions


def _build_raw_actions(
    events: pd.DataFrame,
    preserve_native: list[str] | None,
) -> pd.DataFrame:
    """Build raw SPADL actions from Metrica event rows.

    Implements the §6.3 dispatch + §6.4 set-piece-then-shot composition rule.
    Direction-of-play flipping happens later in ``convert_to_actions``;
    this function emits source-orientation coords.
    """
    mask = events["type"].isin(_MAPPED_TYPES)
    rows = events.loc[mask].copy().reset_index(drop=True)
    n = len(rows)

    if n == 0:
        return _empty_raw_actions(preserve_native, events)

    type_ids = np.zeros(n, dtype=np.int64)
    result_ids = np.full(n, spadlconfig.result_id["success"], dtype=np.int64)
    bodypart_ids = np.full(n, spadlconfig.bodypart_id["foot"], dtype=np.int64)

    typ = rows["type"].to_numpy()
    sub_raw = rows["subtype"].fillna("").astype(str).str.upper().to_numpy()

    # --- PASS ---
    is_pass = typ == "PASS"
    is_pass_cross = is_pass & (sub_raw == "CROSS")
    is_pass_goalkick = is_pass & (sub_raw == "GOAL KICK")
    is_pass_head = is_pass & (sub_raw == "HEAD")
    is_pass_default = is_pass & ~is_pass_cross & ~is_pass_goalkick
    type_ids[is_pass_default] = spadlconfig.actiontype_id["pass"]
    type_ids[is_pass_cross] = spadlconfig.actiontype_id["cross"]
    type_ids[is_pass_goalkick] = spadlconfig.actiontype_id["goalkick"]
    bodypart_ids[is_pass_head] = spadlconfig.bodypart_id["head"]

    # --- RECOVERY -> interception ---
    is_recovery = typ == "RECOVERY"
    type_ids[is_recovery] = spadlconfig.actiontype_id["interception"]

    # --- CHALLENGE WON -> tackle; LOST/AERIAL-* -> drop ---
    is_challenge = typ == "CHALLENGE"
    is_challenge_won = is_challenge & (sub_raw == "WON")
    type_ids[is_challenge_won] = spadlconfig.actiontype_id["tackle"]
    is_challenge_dropped = is_challenge & ~is_challenge_won
    type_ids[is_challenge_dropped] = spadlconfig.actiontype_id["non_action"]

    # --- BALL LOST -> bad_touch fail ---
    is_ball_lost = typ == "BALL LOST"
    type_ids[is_ball_lost] = spadlconfig.actiontype_id["bad_touch"]
    result_ids[is_ball_lost] = spadlconfig.result_id["fail"]

    # --- FAULT -> foul fail (CARD pairing applied below) ---
    is_fault = typ == "FAULT"
    type_ids[is_fault] = spadlconfig.actiontype_id["foul"]
    result_ids[is_fault] = spadlconfig.result_id["fail"]

    # --- SET PIECE dispatch on subtype ---
    is_setpiece = typ == "SET PIECE"
    is_sp_freekick = is_setpiece & (sub_raw == "FREE KICK")
    is_sp_corner = is_setpiece & (sub_raw == "CORNER KICK")
    is_sp_throwin = is_setpiece & (sub_raw == "THROW IN")
    is_sp_goalkick = is_setpiece & (sub_raw == "GOAL KICK")
    is_sp_kickoff = is_setpiece & (sub_raw == "KICK OFF")
    type_ids[is_sp_freekick] = spadlconfig.actiontype_id["freekick_short"]
    type_ids[is_sp_corner] = spadlconfig.actiontype_id["corner_short"]
    type_ids[is_sp_throwin] = spadlconfig.actiontype_id["throw_in"]
    bodypart_ids[is_sp_throwin] = spadlconfig.bodypart_id["other"]
    type_ids[is_sp_goalkick] = spadlconfig.actiontype_id["goalkick"]
    type_ids[is_sp_kickoff] = spadlconfig.actiontype_id["non_action"]

    # --- SHOT (with set-piece composition) ---
    is_shot = typ == "SHOT"
    type_ids[is_shot] = spadlconfig.actiontype_id["shot"]
    is_goal = is_shot & (sub_raw == "GOAL")
    result_ids[is_shot] = spadlconfig.result_id["fail"]
    result_ids[is_goal] = spadlconfig.result_id["success"]

    # §6.4 composition: SET PIECE FREE KICK + SHOT (≤ 5s, same player, same period)
    # → upgrade SHOT to shot_freekick AND drop the SET PIECE row.
    drop_mask = np.zeros(n, dtype=bool)
    if is_sp_freekick.any() and is_shot.any():
        for i in np.where(is_sp_freekick)[0]:
            if i + 1 >= n:
                continue
            cur_row = rows.iloc[i]
            next_row = rows.iloc[i + 1]
            if (
                typ[i + 1] == "SHOT"
                and next_row["period"] == cur_row["period"]
                and next_row["player"] == cur_row["player"]
                and (next_row["start_time_s"] - cur_row["start_time_s"]) <= 5.0
            ):
                type_ids[i + 1] = spadlconfig.actiontype_id["shot_freekick"]
                drop_mask[i] = True

    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["start_time_s"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player"].astype("object"),
            "start_x": rows["start_x"].astype(np.float64),
            "start_y": rows["start_y"].astype(np.float64),
            "end_x": rows["end_x"].astype(np.float64),
            "end_y": rows["end_y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
        }
    )

    actions = actions.loc[~drop_mask]
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)

    # If post-filter all rows dropped, return canonical empty schema so downstream
    # _finalize_output can assemble action_id and other schema-required columns.
    if len(actions) == 0:
        return _empty_raw_actions(preserve_native, events)

    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows.loc[actions.index, col].to_numpy()
            else:
                actions[col] = np.nan

    actions = _apply_card_pairs(actions, events)

    return actions
