"""Sportec (DFL) DataFrame SPADL converter.

Converts already-normalized DFL event DataFrames (e.g., luxury-lakehouse
``bronze.idsse_events`` shape, Bassek 2025 DFL parse output) to SPADL actions.

Consumers with raw DFL XML files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.sportec.load_event(...)``.
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances, _fix_direction_of_play
from .schema import KLOPPY_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns, _validate_preserve_native

# ---------------------------------------------------------------------------
# Required input columns (raise ValueError if any are missing)
# ---------------------------------------------------------------------------
EXPECTED_INPUT_COLUMNS: set[str] = {
    "match_id",
    "event_id",
    "event_type",
    "period",
    "timestamp_seconds",
    "player_id",
    "team",
    "x",
    "y",
}

# ---------------------------------------------------------------------------
# DFL event_type -> SPADL action dispatch
# ---------------------------------------------------------------------------
_MAPPED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "Pass",
        "ShotAtGoal",
        "TacklingGame",
        "Foul",
        "FreeKick",
        "Corner",
        "ThrowIn",
        "GoalKick",
        "Play",
    }
)

_EXCLUDED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "Substitution",
        "Caution",
        "Whistle",
        "Offside",
        "KickOff",
        "OtherBallContact",
        "OtherPlayerAction",
        "Delete",
        "RefereeBall",
        "FairPlay",
        "PlayerOff",
        "PlayerOn",
    }
)

# ---------------------------------------------------------------------------
# Recognized DFL qualifier columns (optional — converter accepts these silently)
# ---------------------------------------------------------------------------
# Sourced from luxury-lakehouse's ``bronze.idsse_events`` schema (the most
# comprehensive normalized DFL shape currently in production). Any subset may
# appear on the input DataFrame; the converter consults a subset of these
# (see ``_CONSULTED_QUALIFIER_COLUMNS``) and silently ignores the rest.
# Unknown columns NOT in this set are also ignored unless listed in
# ``preserve_native``, in which case they pass through to the output.
_RECOGNIZED_QUALIFIER_COLUMNS: frozenset[str] = frozenset(
    {
        # Pass qualifiers
        "pass_direction",
        "pass_free_kick_layup",
        "pass_one_two",
        # Shot qualifiers (DFL has very rich shot annotation)
        "shot_team",
        "shot_player",
        "shot_pressure",
        "shot_inside_box",
        "shot_distance_to_goal",
        "shot_angle_to_goal",
        "shot_type_of_shot",
        "shot_extended_type_of_shot",
        "shot_outcome_type",
        "shot_outcome_current_result",
        "shot_outcome_save_result",
        "shot_outcome_save_evaluation",
        "shot_outcome_save_type",
        "shot_outcome_goal_keeper",
        "shot_outcome_assist",
        "shot_outcome_assist_type",
        "shot_outcome_solo",
        "shot_outcome_goal_zone",
        "shot_outcome_placing",
        "shot_outcome_pitch_marking",
        "shot_outcome_player",
        "shot_outcome_goal_prevented",
        "shot_outcome_location",
        "shot_outcome_blocked_by_own_team",
        "shot_outcome_error",
        "shot_outcome_deflection_keeper",
        "shot_outcome_deflection_player",
        "shot_outcome_assist_contribution",
        "shot_outcome_assist_fouled_player",
        "shot_outcome_ref_decision_evaluation",
        "shot_after_free_kick",
        "shot_assist_action",
        "shot_assist_shot_at_goal",
        "shot_assist_type_shot_at_goal",
        "shot_amount_of_defenders",
        "shot_ball_possession_phase",
        "shot_build_up",
        "shot_chance_evaluation",
        "shot_counter_attack",
        "shot_direct_free_kick_intention",
        "shot_goal_distance_goalkeeper",
        "shot_player_speed",
        "shot_setup_origin",
        "shot_taker_setup",
        "shot_taker_ball_control",
        "shot_significance_evaluation",
        "shot_shot_origin",
        "shot_shot_condition",
        "shot_shot_contribution",
        "shot_shot_assist_fouled_player",
        "shot_sitter_contribution",
        "shot_x_g",
        "shot_rotation",
        "shot_penalty_direction",
        "shot_penalty_execution",
        # Tackle qualifiers
        "tackle_winner",
        "tackle_winner_team",
        "tackle_winner_role",
        "tackle_winner_action",
        "tackle_winner_result",
        "tackle_loser",
        "tackle_loser_team",
        "tackle_loser_role",
        "tackle_type",
        "tackle_possession_change",
        "tackle_goal_keeper_involved",
        "tackle_ball_possession_phase",
        "tackle_dribble_evaluation",
        "tackle_dribbling_side",
        "tackle_dribbling_type",
        # Foul / set-piece anchors
        "foul_team_fouler",
        "foul_fouler",
        "foul_fouled",
        "foul_team_fouled",
        "foul_foul_type",
        "foul_committing_player_action",
        "freekick_team",
        "freekick_decision_timestamp",
        "freekick_execution_mode",
        "corner_team",
        "corner_side",
        "corner_placing",
        "corner_decision_timestamp",
        "corner_rotation",
        "corner_post_marking",
        "corner_target_area",
        "throwin_team",
        "throwin_side",
        "throwin_decision_timestamp",
        "goalkick_team",
        "goalkick_decision_timestamp",
        # Play / general action context
        "play_player",
        "play_team",
        "play_play_origin",
        "play_play_angle",
        "play_recipient",
        "play_ball_possession_phase",
        "play_evaluation",
        "play_goal_keeper_action",
        "play_height",
        "play_distance",
        "play_from_open_play",
        "play_penalty_box",
        "play_flat_cross",
        "play_rotation",
        "play_semi_field",
        "otherball_player",
        "otherball_team",
        "otherball_ball_possession_phase",
        "otherball_defensive_clearance",
        "cross_side",
        "cross_goal_keeper",
        "cross_goal_keeper_interference",
        # Cards / discipline
        "caution_player",
        "caution_team",
        "caution_card_color",
        "caution_official_card_color",
        "caution_reason",
        "caution_other_reason",
        "caution_card_rating",
        "caution_official_team",
        "caution_official_person_sent_off",
        "caution_ref_decision_evaluation",
        # Substitution / player flow
        "sub_player_in",
        "sub_player_out",
        "sub_team",
        "sub_playing_position",
        "other_action_player",
        "other_action_team",
        "other_action_player_becomes_goalkeeper",
        "other_action_change_of_captain",
        "other_action_change_contingent_exhausted",
        # Penalty / VAR / chance / specialised
        "penalty_team",
        "penalty_causing_player",
        "penalty_decision_timestamp",
        "penalty_prospective_taker",
        "penalty_fouled_player",
        "penalty_players_in_box",
        "penalty_goalkeeper_behaviour",
        "penalty_goalkeeper_movement",
        "penalty_retaken_penalty",
        "penalty_ref_decision_evaluation",
        "penalty_not_team",
        "penalty_not_causing_player",
        "penalty_not_player_to_be_awarded",
        "penalty_not_reason",
        "penalty_not_ref_decision_evaluation",
        "not_sent_off_player",
        "not_sent_off_team",
        "not_sent_off_reason",
        "not_sent_off_type",
        "not_sent_off_ref_decision_evaluation",
        "chance_player",
        "chance_team",
        "chance_situation",
        "chance_sitter",
        "chance_setup_origin",
        "chance_taker_setup",
        "chance_chance_assist",
        "chance_chance_assist_type",
        "chance_assist_action",
        "chance_counter_attack",
        "chance_prevention_goalkeeper",
        "goaldis_player",
        "goaldis_team",
        "goaldis_reason",
        "goaldis_ref_decision_evaluation",
        "run_player",
        "run_team",
        "spectacular_player",
        "spectacular_team",
        "spectacular_type",
        "possloss_player",
        "possloss_team",
        "possloss_type_of_possession_loss",
        "possloss_possession_loss_origin",
        "sitter_prev_player",
        "sitter_prev_team",
        "sitter_prev_reason",
        "sitter_prev_ref_decision_evaluation",
        "deflection_player",
        "deflection_team",
        "deflection_type",
        "claim_player",
        "claim_team",
        "claim_type",
        "claim_ball_possession_phase",
        "fault_execution_player",
        "fault_execution_team",
        "fault_execution_ball_possession_phase",
        "nutmeg_player",
        "nutmeg_team",
        "nutmeg_affected_player",
        "nutmeg_affected_team",
        "fairplay_player",
        "fairplay_team",
        "fairplay_ball_possession_phase",
        "offside_player",
        "offside_team",
        "delete_reason",
        "var_team_challenged",
        "var_proofed_event",
        "var_referee",
        "var_video_assistant",
        "var_linesman1",
        "var_linesman2",
        "var_refereein_rra",
        "var_opponent_team",
        "var_ref_decision",
        "var_ref_decision_evaluation",
        "var_final_decision",
        "var_timestamp_start_action",
        "var_timestamp_end_action",
        "whistle_game_section",
        "whistle_final_result",
        "whistle_breaking_off",
        # Tracking-derived position fields
        "x_source_position",
        "y_source_position",
        "x_position_from_tracking",
        "y_position_from_tracking",
        "calculated_frame",
        "calculated_timestamp",
        "start_frame",
        "end_frame",
    }
)

# Subset of recognized qualifiers that the converter actually consults to
# refine SPADL action type / result / bodypart. Documented separately so
# downstream consumers can see at a glance which DFL qualifiers actually
# influence the SPADL output (vs which are merely tolerated as input).
_CONSULTED_QUALIFIER_COLUMNS: frozenset[str] = frozenset(
    {
        "play_height",
        "play_flat_cross",
        "freekick_execution_mode",
        "corner_target_area",
        "corner_placing",
        "shot_after_free_kick",
        "shot_outcome_type",
        "penalty_team",
        "penalty_causing_player",
        "tackle_winner",
        "tackle_winner_team",
        "foul_fouler",
        "caution_player",
        "caution_card_color",
        "play_goal_keeper_action",
    }
)


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert normalized Sportec/DFL event DataFrame to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Normalized DFL event data with columns per ``EXPECTED_INPUT_COLUMNS``
        plus any subset of optional qualifier columns.
    home_team_id : str
        Identifier of the home team (used to flip away-team coords for canonical
        SPADL "all-actions-LTR" orientation).
    preserve_native : list[str], optional
        Caller-attached columns to preserve through to the output. Synthetic
        dribble rows get NaN in preserved columns.

    Returns
    -------
    actions : pd.DataFrame
        SPADL actions with ``KLOPPY_SPADL_COLUMNS`` schema plus preserved extras.
    report : ConversionReport
        Audit trail with mapped/excluded/unrecognized event-type counts.
    """
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Sportec")
    _validate_preserve_native(events, preserve_native, provider="Sportec", schema=KLOPPY_SPADL_COLUMNS)

    event_type_counts = Counter(events["event_type"])

    raw_actions = _build_raw_actions(events, preserve_native)

    if len(raw_actions) > 0:
        actions = _fix_clearances(raw_actions)
        actions = _fix_direction_of_play(actions, home_team_id)
        actions["action_id"] = range(len(actions))
        actions = _add_dribbles(actions)

        # Clamp coords to SPADL pitch frame (consistent with 1.6.0 kloppy converter).
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
        if et in _MAPPED_EVENT_TYPES:
            mapped_counts[label] = count
        elif et in _EXCLUDED_EVENT_TYPES:
            excluded_counts[label] = count
        else:
            unrecognized_counts[label] = count
    if unrecognized_counts:
        warnings.warn(
            f"Sportec: {sum(unrecognized_counts.values())} unrecognized event types "
            f"dropped: {dict(unrecognized_counts)}",
            stacklevel=2,
        )
    report = ConversionReport(
        provider="Sportec",
        total_events=sum(event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
    )
    return actions, report


def _empty_raw_actions(preserve_native: list[str] | None, events: pd.DataFrame) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical SPADL schema plus extras."""
    cols = {col: pd.Series(dtype=dtype) for col, dtype in KLOPPY_SPADL_COLUMNS.items()}
    if preserve_native:
        for col in preserve_native:
            cols[col] = pd.Series(dtype=events[col].dtype if col in events.columns else "object")
    return pd.DataFrame(cols)


def _find_caution_pairs(
    events: pd.DataFrame,
    rows: pd.DataFrame,
    is_foul: np.ndarray,
) -> dict[int, str]:
    """Find Foul rows paired with subsequent Caution rows (same fouler, ≤ 3s)."""
    pairs: dict[int, str] = {}
    if "caution_card_color" not in events.columns:
        return pairs
    foul_indices = np.where(is_foul)[0]
    for idx in foul_indices:
        foul_row = rows.iloc[idx]
        candidates = events[
            (events["event_type"] == "Caution")
            & (events["period"] == foul_row["period"])
            & (events["timestamp_seconds"] >= foul_row["timestamp_seconds"])
            & (events["timestamp_seconds"] <= foul_row["timestamp_seconds"] + 3.0)
        ]
        if candidates.empty:
            continue
        foul_fouler = rows.get("foul_fouler", pd.Series([None])).iloc[idx] if "foul_fouler" in rows.columns else None
        if foul_fouler is not None and not pd.isna(foul_fouler) and "caution_player" in candidates.columns:
            matched = candidates[candidates["caution_player"] == foul_fouler]
        else:
            matched = candidates[candidates["player_id"] == foul_row["player_id"]]
        if not matched.empty:
            pairs[idx] = str(matched["caution_card_color"].iloc[0])
    return pairs


def _build_raw_actions(
    events: pd.DataFrame,
    preserve_native: list[str] | None,
) -> pd.DataFrame:
    """Build raw SPADL actions DataFrame from DFL event rows.

    For each row, dispatches on ``event_type`` to determine SPADL
    type/result/bodypart. Drops rows whose ``event_type`` is in
    ``_EXCLUDED_EVENT_TYPES`` or not in ``_MAPPED_EVENT_TYPES``.
    Direction-of-play flipping happens later in ``convert_to_actions``;
    this function emits source-orientation coords.
    """
    mask = events["event_type"].isin(_MAPPED_EVENT_TYPES)
    rows = events.loc[mask].copy().reset_index(drop=True)
    n = len(rows)

    if n == 0:
        return _empty_raw_actions(preserve_native, events)

    type_ids = np.zeros(n, dtype=np.int64)
    result_ids = np.zeros(n, dtype=np.int64)
    bodypart_ids = np.full(n, spadlconfig.bodypart_id["foot"], dtype=np.int64)

    et = rows["event_type"].to_numpy()

    def _opt(col: str, default):
        """Return rows[col] if present, else a Series of `default` of length n."""
        if col in rows.columns:
            return rows[col]
        return pd.Series([default] * n, index=rows.index)

    # --- Pass / Cross detection ---
    is_pass = et == "Pass"
    is_cross_by_height = is_pass & _opt("play_height", None).eq("cross").to_numpy()
    is_cross_by_flag = is_pass & _opt("play_flat_cross", False).fillna(False).astype(bool).to_numpy()
    is_cross = is_cross_by_height | is_cross_by_flag
    is_pass_plain = is_pass & ~is_cross
    type_ids[is_pass_plain] = spadlconfig.actiontype_id["pass"]
    type_ids[is_cross] = spadlconfig.actiontype_id["cross"]
    result_ids[is_pass] = spadlconfig.result_id["success"]

    # Pass head bodypart
    is_head = is_pass & _opt("play_height", None).eq("head").to_numpy()
    bodypart_ids[is_head] = spadlconfig.bodypart_id["head"]

    # --- Set pieces ---
    is_freekick = et == "FreeKick"
    fk_exec = _opt("freekick_execution_mode", "").fillna("").to_numpy()
    is_fk_crossed = is_freekick & np.isin(fk_exec, ["cross", "long", "longBall", "highPass"])
    type_ids[is_freekick & ~is_fk_crossed] = spadlconfig.actiontype_id["freekick_short"]
    type_ids[is_fk_crossed] = spadlconfig.actiontype_id["freekick_crossed"]
    result_ids[is_freekick] = spadlconfig.result_id["success"]

    is_corner = et == "Corner"
    corner_target = _opt("corner_target_area", "").fillna("").to_numpy()
    corner_placing = _opt("corner_placing", "").fillna("").to_numpy()
    is_corner_crossed = is_corner & (
        np.isin(corner_target, ["box", "penaltyBox", "sixYardBox"]) | np.isin(corner_placing, ["aerial", "high"])
    )
    type_ids[is_corner & ~is_corner_crossed] = spadlconfig.actiontype_id["corner_short"]
    type_ids[is_corner_crossed] = spadlconfig.actiontype_id["corner_crossed"]
    result_ids[is_corner] = spadlconfig.result_id["success"]

    is_throwin = et == "ThrowIn"
    type_ids[is_throwin] = spadlconfig.actiontype_id["throw_in"]
    bodypart_ids[is_throwin] = spadlconfig.bodypart_id["other"]
    result_ids[is_throwin] = spadlconfig.result_id["success"]

    is_goalkick = et == "GoalKick"
    type_ids[is_goalkick] = spadlconfig.actiontype_id["goalkick"]
    result_ids[is_goalkick] = spadlconfig.result_id["success"]

    # --- Shot ---
    is_shot = et == "ShotAtGoal"
    after_fk = _opt("shot_after_free_kick", "").fillna("").astype(str).str.lower().eq("true").to_numpy()
    has_penalty = (_opt("penalty_team", None).notna() | _opt("penalty_causing_player", None).notna()).to_numpy()
    is_shot_penalty = is_shot & has_penalty
    is_shot_freekick = is_shot & after_fk & ~is_shot_penalty
    is_shot_plain = is_shot & ~is_shot_penalty & ~is_shot_freekick
    type_ids[is_shot_plain] = spadlconfig.actiontype_id["shot"]
    type_ids[is_shot_freekick] = spadlconfig.actiontype_id["shot_freekick"]
    type_ids[is_shot_penalty] = spadlconfig.actiontype_id["shot_penalty"]

    shot_outcome = _opt("shot_outcome_type", "").fillna("").astype(str).to_numpy()
    is_goal = is_shot & (shot_outcome == "goal")
    is_owngoal = is_shot & (shot_outcome == "ownGoal")
    result_ids[is_shot] = spadlconfig.result_id["fail"]
    result_ids[is_goal] = spadlconfig.result_id["success"]
    # Owngoals: emit as bad_touch + result=owngoal per existing converter precedent.
    type_ids[is_owngoal] = spadlconfig.actiontype_id["bad_touch"]
    result_ids[is_owngoal] = spadlconfig.result_id["owngoal"]

    # --- TacklingGame: actor = tackle_winner if present, else generic player_id/team ---
    is_tackle = et == "TacklingGame"
    type_ids[is_tackle] = spadlconfig.actiontype_id["tackle"]
    result_ids[is_tackle] = spadlconfig.result_id["success"]
    if is_tackle.any():
        winner_p = _opt("tackle_winner", None)
        winner_t = _opt("tackle_winner_team", None)
        override_mask = is_tackle & winner_p.notna().to_numpy()
        if override_mask.any():
            rows.loc[override_mask, "player_id"] = winner_p[override_mask].values
            rows.loc[override_mask, "team"] = winner_t[override_mask].values

    # --- Foul (with Caution pairing for card upgrade) ---
    is_foul = et == "Foul"
    type_ids[is_foul] = spadlconfig.actiontype_id["foul"]
    result_ids[is_foul] = spadlconfig.result_id["fail"]

    if is_foul.any():
        pairs = _find_caution_pairs(events, rows, is_foul)
        for idx, paired_color in pairs.items():
            if paired_color == "yellow":
                result_ids[idx] = spadlconfig.result_id["yellow_card"]
            elif paired_color == "secondYellow":
                result_ids[idx] = spadlconfig.result_id["red_card"]
            elif paired_color in ("red", "directRed"):
                result_ids[idx] = spadlconfig.result_id["red_card"]

    # --- Play with goalkeeper_action ---
    is_play = et == "Play"
    play_gk = _opt("play_goal_keeper_action", "").fillna("").astype(str).to_numpy()
    play_gk_save = is_play & (play_gk == "save")
    play_gk_claim = is_play & (play_gk == "claim")
    play_gk_punch = is_play & (play_gk == "punch")
    play_gk_pickup = is_play & (play_gk == "pickUp")
    type_ids[play_gk_save] = spadlconfig.actiontype_id["keeper_save"]
    type_ids[play_gk_claim] = spadlconfig.actiontype_id["keeper_claim"]
    type_ids[play_gk_punch] = spadlconfig.actiontype_id["keeper_punch"]
    type_ids[play_gk_pickup] = spadlconfig.actiontype_id["keeper_pick_up"]
    result_ids[is_play] = spadlconfig.result_id["success"]

    is_play_no_action = is_play & ~(play_gk_save | play_gk_claim | play_gk_punch | play_gk_pickup)
    type_ids[is_play_no_action] = spadlconfig.actiontype_id["non_action"]

    # Assemble actions DataFrame.
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["timestamp_seconds"].astype(np.float64),
            "team_id": rows["team"].astype("object"),
            "player_id": rows["player_id"].astype("object"),
            "start_x": rows["x"].astype(np.float64),
            "start_y": rows["y"].astype(np.float64),
            "end_x": rows["x"].astype(np.float64),
            "end_y": rows["y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
        }
    )

    # Drop non_action rows.
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)

    # If post-filter all rows dropped, return canonical empty schema so downstream
    # _finalize_output can assemble action_id and other schema-required columns.
    if len(actions) == 0:
        return _empty_raw_actions(preserve_native, events)

    # Carry preserve_native columns through.
    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows.loc[actions.index, col].to_numpy()
            else:
                actions[col] = np.nan

    return actions
