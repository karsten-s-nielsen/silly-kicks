"""Sportec (DFL) DataFrame SPADL converter.

Converts already-normalized DFL event DataFrames (e.g., luxury-lakehouse
``bronze.idsse_events`` shape, Bassek 2025 DFL parse output) to SPADL actions.

Consumers with raw DFL XML files should use ``silly_kicks.spadl.kloppy`` after
``kloppy.sportec.load_event(...)``.

DFL ``event_type`` vocabulary (recognized as input)
---------------------------------------------------

The dispatch consumes the following DFL event_type values (case-sensitive):

================== ======================================================
``event_type``     SPADL mapping
================== ======================================================
``Play``           Pass-class by default; refined by qualifier (see below)
``ShotAtGoal``     ``shot`` / ``shot_freekick`` / ``shot_penalty``
``TacklingGame``   ``tackle``
``Foul``           ``foul`` (with optional ``Caution`` pairing for cards)
``FreeKick``       ``freekick_short`` / ``freekick_crossed``
``Corner``         ``corner_short`` / ``corner_crossed``
``ThrowIn``        ``throw_in``
``GoalKick``       ``goalkick``
================== ======================================================

DFL ``play_goal_keeper_action`` qualifier vocabulary
----------------------------------------------------

For ``Play`` events, the ``play_goal_keeper_action`` qualifier disambiguates
between pass-class and GK-class semantics:

================== ============================== ============================
Qualifier value    SPADL mapping                  Notes
================== ============================== ============================
``""`` (empty)     ``pass`` / ``cross``           Pass-class default
``save``           ``keeper_save``
``claim``          ``keeper_claim``
``punch``          ``keeper_punch``
``pickUp``         ``keeper_pick_up``
``throwOut``       ``keeper_pick_up`` + ``pass``  GK distribution by hand
``punt``           ``keeper_pick_up`` + ``goalkick``  GK distribution by foot
unrecognized       ``non_action`` (filtered)      Defensive
================== ============================== ============================

The ``throwOut`` and ``punt`` rows synthesize TWO SPADL actions per source
event: a ``keeper_pick_up`` representing the GK's reception of the ball,
followed by a ``pass`` (for ``throwOut``) or ``goalkick`` (for ``punt``)
representing the GK's distribution. Action_ids are renumbered dense after
synthesis. ``preserve_native`` columns propagate to both rows.

Bug history
-----------

Pre-1.10.0:

- ``is_pass = et == "Pass"`` silently dropped ALL DFL ``Play`` events
  (the actual pass-class event_type) to ``non_action``. Net effect: all
  IDSSE matches in production lost ALL pass-class events for ~3 release
  cycles. Fixed in 1.10.0 by restructuring the dispatch around
  ``et == "Play"``.
- ``play_goal_keeper_action`` qualifier vocabulary was incomplete (only
  ``save`` / ``claim`` / ``punch`` / ``pickUp``); ``throwOut`` and ``punt``
  silently dropped to ``non_action``. Fixed in 1.10.0 by adding the
  2-action synthesis.

See ``docs/superpowers/specs/2026-04-29-gk-converter-coverage-parity-design.md``
for the full design rationale.

ADR-001: identifier conventions are sacred (silly-kicks 2.0.0)
---------------------------------------------------------------

The converter never overrides ``team_id`` / ``player_id`` from DFL
qualifier columns. Caller-supplied ``team`` / ``player_id`` values
mirror verbatim into the output. DFL ``tackle_winner`` /
``tackle_winner_team`` / ``tackle_loser`` / ``tackle_loser_team``
qualifier values surface via dedicated output columns:

==========================  ============================  ============================
Output column               DFL qualifier source          NaN when
==========================  ============================  ============================
``tackle_winner_player_id`` ``tackle_winner``             qualifier absent OR non-tackle row
``tackle_winner_team_id``   ``tackle_winner_team``        qualifier absent OR non-tackle row
``tackle_loser_player_id``  ``tackle_loser``              qualifier absent OR non-tackle row
``tackle_loser_team_id``    ``tackle_loser_team``         qualifier absent OR non-tackle row
==========================  ============================  ============================

The output schema is :data:`silly_kicks.spadl.SPORTEC_SPADL_COLUMNS`
(extends :data:`silly_kicks.spadl.KLOPPY_SPADL_COLUMNS` with the 4
tackle columns).

Pre-2.0.0 callers relying on the SPADL "tackle.actor = winner" semantic —
specifically those whose upstream-supplied ``team`` / ``player_id``
columns happened to be in the same identifier convention as DFL's
``tackle_winner_team`` / ``tackle_winner`` qualifiers (raw
``DFL-CLU-...`` / ``DFL-OBJ-...``) — restore the prior behavior with the
:func:`silly_kicks.spadl.use_tackle_winner_as_actor` helper:

.. code-block:: python

    from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
    actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
    actions = use_tackle_winner_as_actor(actions)

See ``docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md``
for the contract rationale and audit findings.
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import config as spadlconfig
from .base import _add_dribbles, _fix_clearances, _fix_direction_of_play
from .schema import KLOPPY_SPADL_COLUMNS, SPORTEC_SPADL_COLUMNS, ConversionReport
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
        # NOTE: "Pass" was in this set pre-1.10.0 but DFL bronze never emits
        # it — the actual pass-class event_type is "Play". Removing "Pass"
        # makes legacy callers' rows surface in unrecognized_counts (loud)
        # rather than silently mapping to non_action.
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
    goalkeeper_ids: set[str] | None = None,
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
    goalkeeper_ids : set[str] or None, default ``None``
        Optional set of DFL player_ids known to be goalkeepers. When
        provided, Play events whose ``player_id`` is in this set and which
        have NO explicit ``play_goal_keeper_action`` qualifier are routed
        to the keeper_pick_up + pass 2-action synthesis (matching the
        ``throwOut`` qualifier shape). Use this to surface GK distribution
        events that DFL didn't natively annotate. The qualifier-driven
        mapping (save / claim / punch / pickUp / throwOut / punt) remains
        the primary contract and takes precedence when both signals are
        present. An empty set is equivalent to ``None``.

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

    raw_actions = _build_raw_actions(events, preserve_native, goalkeeper_ids=goalkeeper_ids)

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
    actions = _finalize_output(actions, schema=SPORTEC_SPADL_COLUMNS, extra_columns=extras)

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
    cols = {col: pd.Series(dtype=dtype) for col, dtype in SPORTEC_SPADL_COLUMNS.items()}
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
    *,
    goalkeeper_ids: set[str] | None = None,
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

    # --- DFL "Play" events: pass-class by default; GK action when qualifier present ---
    # DFL bronze never emits "Pass" — the event_type for pass-class events is "Play".
    # Pre-1.10.0 the dispatch checked et == "Pass", silently dropping every IDSSE
    # pass to non_action. Post-1.10.0:
    #   - Play with empty/null qualifier   → pass-class (pass / cross / head bodypart)
    #   - Play with save/claim/punch/pickUp → keeper_save / keeper_claim / keeper_punch / keeper_pick_up
    #   - Play with throwOut → keeper_pick_up + synthesized "pass"  (distribution by hand)
    #   - Play with punt → keeper_pick_up + synthesized "goalkick"  (distribution by foot)
    #   - Play with unrecognized non-empty qualifier → non_action (defensive)
    is_play = et == "Play"
    play_gk = _opt("play_goal_keeper_action", "").fillna("").astype(str).to_numpy()
    play_gk_save = is_play & (play_gk == "save")
    play_gk_claim = is_play & (play_gk == "claim")
    play_gk_punch = is_play & (play_gk == "punch")
    play_gk_pickup = is_play & (play_gk == "pickUp")
    play_gk_throwout = is_play & (play_gk == "throwOut")
    play_gk_punt = is_play & (play_gk == "punt")
    play_gk_distribution = play_gk_throwout | play_gk_punt
    is_play_known_qualifier = play_gk_save | play_gk_claim | play_gk_punch | play_gk_pickup | play_gk_distribution

    # Supplementary signal: when goalkeeper_ids is provided, Play events
    # with player_id in the set AND no native GK qualifier route to the
    # keeper_pick_up + pass synthesis. The qualifier-driven path takes
    # precedence (no overlap by construction — supplementary fires only
    # when play_gk == "").
    play_gk_supplementary = np.zeros(n, dtype=bool)
    if goalkeeper_ids:
        gk_str_set = {str(g) for g in goalkeeper_ids}
        is_known_gk_player = rows["player_id"].astype(str).isin(gk_str_set).to_numpy()
        play_gk_supplementary = is_play & (play_gk == "") & is_known_gk_player

    is_play_gk_any = is_play_known_qualifier | play_gk_supplementary

    # Pass-class: Play with empty qualifier AND not in supplementary path.
    is_pass = is_play & (play_gk == "") & ~play_gk_supplementary
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

    # --- TacklingGame ---
    # Per ADR-001: caller's team / player_id mirror verbatim; the converter
    # never overrides them from DFL qualifier columns. Winner / loser ids
    # surface as dedicated tackle_*_player_id / tackle_*_team_id output
    # columns below. Callers wanting the SPADL-canonical "actor = winner"
    # semantic apply silly_kicks.spadl.use_tackle_winner_as_actor() post-conversion.
    is_tackle = et == "TacklingGame"
    type_ids[is_tackle] = spadlconfig.actiontype_id["tackle"]
    result_ids[is_tackle] = spadlconfig.result_id["success"]

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

    # --- GK action assignment (Play events with recognized qualifier) ---
    # is_play / play_gk / play_gk_* masks computed in the consolidated Play
    # dispatch block above. Distribution rows (throwOut/punt) and the
    # supplementary path get keeper_pick_up here; the synthesis helper
    # below adds their second half (pass / goalkick).
    type_ids[play_gk_save] = spadlconfig.actiontype_id["keeper_save"]
    type_ids[play_gk_claim] = spadlconfig.actiontype_id["keeper_claim"]
    type_ids[play_gk_punch] = spadlconfig.actiontype_id["keeper_punch"]
    type_ids[play_gk_pickup] = spadlconfig.actiontype_id["keeper_pick_up"]
    type_ids[play_gk_distribution] = spadlconfig.actiontype_id["keeper_pick_up"]
    type_ids[play_gk_supplementary] = spadlconfig.actiontype_id["keeper_pick_up"]
    bodypart_ids[is_play_gk_any] = spadlconfig.bodypart_id["other"]
    result_ids[is_play_gk_any] = spadlconfig.result_id["success"]

    # Play events with UNRECOGNIZED non-empty qualifier value → non_action
    # (defensive: avoids silently emitting pass for future qualifier values
    # we haven't analyzed). Empty-qualifier rows are pass-class above.
    is_play_unrecognized_gk = is_play & (play_gk != "") & ~is_play_known_qualifier
    type_ids[is_play_unrecognized_gk] = spadlconfig.actiontype_id["non_action"]

    # --- ADR-001 qualifier passthrough columns for TacklingGame rows ---
    # Surface DFL tackle_winner / tackle_winner_team / tackle_loser /
    # tackle_loser_team verbatim, NaN on non-tackle rows. The np.where
    # masks out qualifier values from non-tackle rows defensively (in
    # case caller leaves these columns populated on unrelated events).
    tackle_winner_player_arr = np.where(is_tackle, _opt("tackle_winner", np.nan).to_numpy(dtype=object), np.nan).astype(
        object
    )
    tackle_winner_team_arr = np.where(
        is_tackle, _opt("tackle_winner_team", np.nan).to_numpy(dtype=object), np.nan
    ).astype(object)
    tackle_loser_player_arr = np.where(is_tackle, _opt("tackle_loser", np.nan).to_numpy(dtype=object), np.nan).astype(
        object
    )
    tackle_loser_team_arr = np.where(
        is_tackle, _opt("tackle_loser_team", np.nan).to_numpy(dtype=object), np.nan
    ).astype(object)

    # Assemble main actions DataFrame (1:1 with rows).
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
            "tackle_winner_player_id": tackle_winner_player_arr,
            "tackle_winner_team_id": tackle_winner_team_arr,
            "tackle_loser_player_id": tackle_loser_player_arr,
            "tackle_loser_team_id": tackle_loser_team_arr,
        }
    )

    # Carry preserve_native columns onto main actions BEFORE synthesis so
    # synthetic rows can pick them up via the helper.
    if preserve_native:
        for col in preserve_native:
            if col in rows.columns:
                actions[col] = rows[col].to_numpy()
            else:
                actions[col] = np.nan

    # Synthesize keeper-distribution second-action rows (throwOut/punt
    # and supplementary goalkeeper_ids path; the latter is zeros until
    # Task 5 lights it up).
    synth = _synthesize_gk_distribution_actions(
        rows,
        play_gk_throwout=play_gk_throwout,
        play_gk_punt=play_gk_punt,
        play_gk_supplementary=play_gk_supplementary,
        preserve_native=preserve_native,
    )
    if len(synth) > 0:
        actions["_row_order"] = np.arange(len(actions), dtype=np.int64) * 2
        combined = pd.concat([actions, synth], ignore_index=True, sort=False)
        actions = (
            combined.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"]).reset_index(drop=True)
        )

    # Drop non_action rows.
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)

    if len(actions) == 0:
        return _empty_raw_actions(preserve_native, events)

    return actions


def _synthesize_gk_distribution_actions(
    rows: pd.DataFrame,
    play_gk_throwout: np.ndarray,
    play_gk_punt: np.ndarray,
    play_gk_supplementary: np.ndarray,
    preserve_native: list[str] | None,
) -> pd.DataFrame:
    """Build the synthetic second-action DataFrame for GK distribution events.

    For each row in ``rows`` matching one of the distribution masks, emit a
    synthetic SPADL action immediately after the corresponding keeper_pick_up
    row (which is assigned in the main ``_build_raw_actions`` dispatch).

    Routing:

    - ``throwOut`` qualifier → ``pass`` (bodypart=other; GK distributed by hand)
    - ``punt`` qualifier → ``goalkick`` (bodypart=foot; GK distributed by foot)
    - ``goalkeeper_ids`` supplementary path (no native qualifier present
      but player_id is a known GK): → ``pass`` (bodypart=other; defaults
      to the throwOut shape)

    All synthetic rows inherit the source row's ``(match_id, event_id,
    period, timestamp_seconds, player_id, team, x, y)``. ``original_event_id``
    is the source's ``event_id`` suffixed with ``"_synth_pass"`` /
    ``"_synth_goalkick"`` to keep IDs unique.

    Returns an empty DataFrame when no distribution / supplementary rows
    exist; otherwise returns a synthetic-rows DataFrame with a ``_row_order``
    column used by the caller to interleave synthetic rows after their
    source rows.
    """
    is_distribution = play_gk_throwout | play_gk_punt | play_gk_supplementary
    if not is_distribution.any():
        return pd.DataFrame()

    src_indices = np.where(is_distribution)[0]
    src = rows.iloc[src_indices].copy().reset_index(drop=True)

    n_synth = len(src)
    type_ids_synth = np.full(n_synth, spadlconfig.actiontype_id["pass"], dtype=np.int64)
    bodypart_ids_synth = np.full(n_synth, spadlconfig.bodypart_id["other"], dtype=np.int64)
    suffix = np.full(n_synth, "_synth_pass", dtype=object)

    # punt → goalkick + foot
    is_punt_synth = play_gk_punt[src_indices]
    type_ids_synth[is_punt_synth] = spadlconfig.actiontype_id["goalkick"]
    bodypart_ids_synth[is_punt_synth] = spadlconfig.bodypart_id["foot"]
    suffix[is_punt_synth] = "_synth_goalkick"

    synth = pd.DataFrame(
        {
            "game_id": src["match_id"].astype("object"),
            "original_event_id": (
                src["event_id"].astype(str) + pd.Series(suffix, index=src.index, dtype="object")
            ).astype("object"),
            "period_id": src["period"].astype(np.int64),
            "time_seconds": src["timestamp_seconds"].astype(np.float64),
            "team_id": src["team"].astype("object"),
            "player_id": src["player_id"].astype("object"),
            "start_x": src["x"].astype(np.float64),
            "start_y": src["y"].astype(np.float64),
            "end_x": src["x"].astype(np.float64),
            "end_y": src["y"].astype(np.float64),
            "type_id": type_ids_synth,
            "result_id": np.full(n_synth, spadlconfig.result_id["success"], dtype=np.int64),
            "bodypart_id": bodypart_ids_synth,
        }
    )

    # _row_order: interleave key. Source rows get index*2; synth rows
    # get the source's index*2 + 1 so synth sorts immediately after.
    synth["_row_order"] = src_indices.astype(np.int64) * 2 + 1

    if preserve_native:
        for col in preserve_native:
            if col in src.columns:
                synth[col] = src[col].to_numpy()
            else:
                synth[col] = np.nan

    return synth


_USE_TACKLE_WINNER_AS_ACTOR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "team_id",
    "player_id",
    "tackle_winner_player_id",
    "tackle_winner_team_id",
)


def use_tackle_winner_as_actor(actions: pd.DataFrame) -> pd.DataFrame:
    """Re-attribute SPADL tackle rows to the winning duelist (pre-2.0.0 semantic).

    Sportec converter output emits ``team_id`` / ``player_id`` mirroring the
    caller's input verbatim per ADR-001. For tackle rows where DFL recorded
    a winner via ``tackle_winner`` / ``tackle_winner_team`` qualifiers, the
    winner ids are surfaced as ``tackle_winner_player_id`` /
    ``tackle_winner_team_id`` columns alongside.

    This helper applies the SPADL-canonical "tackle.actor = winner" semantic
    by overwriting ``player_id`` / ``team_id`` from the winner columns where
    those columns are non-null. Pre-2.0.0 sportec consumers can call it
    post-conversion to restore the prior behavior — but ONLY if their
    upstream-supplied ``team`` column is already in the same identifier
    convention as DFL's ``tackle_winner_team`` qualifier (raw
    ``DFL-CLU-...``). Mismatched conventions are the bug ADR-001 fixes;
    this helper does NOT resolve identifier formats.

    Parameters
    ----------
    actions : pd.DataFrame
        Sportec converter output. Must contain ``team_id``, ``player_id``,
        ``tackle_winner_player_id``, ``tackle_winner_team_id``.

    Returns
    -------
    pd.DataFrame
        A copy of ``actions`` with ``team_id`` and ``player_id`` overwritten
        from the winner columns on rows where those columns are non-null.
        All other columns unchanged.

    Raises
    ------
    ValueError
        If a required column is missing.

    Examples
    --------
    Restore SPADL "actor = winner" semantic on sportec output::

        from silly_kicks.spadl import sportec, use_tackle_winner_as_actor
        actions, _ = sportec.convert_to_actions(events, home_team_id="DFL-CLU-XXX")
        actions = use_tackle_winner_as_actor(actions)
    """
    missing = [c for c in _USE_TACKLE_WINNER_AS_ACTOR_REQUIRED_COLUMNS if c not in actions.columns]
    if missing:
        raise ValueError(
            f"use_tackle_winner_as_actor: actions missing required columns: {sorted(missing)}. "
            f"Got: {sorted(actions.columns)}"
        )

    result = actions.copy()
    if len(result) == 0:
        return result

    winner_player = result["tackle_winner_player_id"]
    winner_team = result["tackle_winner_team_id"]
    # Atomic per-row: only overwrite when BOTH winner columns are non-null
    # (matches DFL's qualifier pairing — both populated together or both
    # absent). Rows with mismatched NaN status (a possibility only with
    # pathological hand-crafted fixtures) are conservatively left unchanged.
    overwrite_mask = winner_player.notna() & winner_team.notna()
    if overwrite_mask.any():
        result.loc[overwrite_mask, "player_id"] = winner_player[overwrite_mask].to_numpy()
        result.loc[overwrite_mask, "team_id"] = winner_team[overwrite_mask].to_numpy()

    return result
