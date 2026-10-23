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
    actions, _ = sportec.convert_to_actions(
        events, home_team_id="DFL-CLU-XXX", home_team_start_left=True,
    )
    actions = use_tackle_winner_as_actor(actions)

See ``docs/superpowers/adrs/ADR-001-converter-identifier-conventions.md``
for the contract rationale and audit findings.
"""

import warnings
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..id_compat import canonical_id, canonical_id_series, ids_match
from ..tracking import direction
from . import config as spadlconfig
from .base import _add_dribbles, _derive_end_coordinates, sort_actions_chronologically
from .orientation import PER_PERIOD_ABSOLUTE, to_spadl_ltr, validate_input_convention
from .schema import KLOPPY_SPADL_COLUMNS, SPORTEC_SPADL_COLUMNS, SportecConversionReport
from .utils import (
    _blocked_flag,
    _ensure_base_signal_columns,
    _finalize_output,
    _validate_input_columns,
    _validate_preserve_native,
)

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
        "OtherBallAction",
        # TF-46 Arm 1 Part 2 §25: the standalone DFL Penalty event (no paired ShotAtGoal). Mapped to
        # shot_penalty; its result is reconciled from the final score (_penalty_results).
        "Penalty",
        # TF-46 Arm 1 Part 2 decision 3: the top-level DFL OwnGoal event -> bad_touch / owngoal. The §25.2
        # step-2 identity needs own goals (final = ShotAtGoal goals + opponent own goals + penalties).
        "OwnGoal",
    }
)

# DFL XML tag aliases: the XML first-child tag name may differ from the
# normalized form used in the dispatch logic below. Callers passing raw
# XML tag names (e.g., lakehouse ingestion) need these aliases resolved
# before the whitelist check. See Bug 3 (3.10.1).
_EVENT_TYPE_ALIASES: dict[str, str] = {
    "CornerKick": "Corner",
}

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
        # BallClaiming produces no 1:1 SPADL row; it is consumed as an F4 annotation
        # (_synthesize_ball_claiming_actions) -- a derived interception / keeper_pick_up row plus a
        # ball_claim_type qualifier (on the partner for InterceptedBall / BallHeld / Block, on the derived
        # row for BallClaimed). Listing it here keeps it out of the "unrecognized, dropped" warning (it is
        # recognized and consumed, just not mapped 1:1). See the TF-46 spec (F4).
        "BallClaiming",
        # TF-46 Arm 1 Part 2 §25.4 (Guard 2): the remaining DFL event types are recognized and either
        # consumed as a qualifier or intentionally dropped, so the unrecognized allow-list is empty.
        # BallDeflection is consumed by a qualifier pass (_route_ball_deflections), the BallClaiming
        # precedent above; the rest are annotation / referee / admin / disallowed-goal records with no
        # 1:1 SPADL action meaning.
        "BallDeflection",  # consumed: woodwork -> shot_woodwork_type; referee / unpaired counted (§25.4)
        "PenaltyNotAwarded",  # referee annotation, no action
        "Nutmeg",  # skill annotation on another action
        "FinalWhistle",  # period / match boundary marker (carries whistle_final_result for Guard 1)
        "SpectacularPlay",  # annotation on another action
        "VideoAssistantAction",  # VAR referee annotation, no action
        "ChanceWithoutShot",  # analytic annotation, no action
        "PossessionLossBeforeGoal",  # analytic annotation, no action
        "Run",  # off-ball movement annotation, no action
        "PlayerNotSentOff",  # referee annotation, no action
        "SitterPrevented",  # analytic annotation, no action
        "CautionTeamofficial",  # bench caution, no player action
        "GoalDisallowed",  # disallowed-goal annotation; the disallowed goal is never counted
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
        # Lakehouse-prefixed variants (lakehouse uses event_type prefix)
        "other_ball_action_defensive_clearance",
        "other_ball_action_ball_possession_phase",
        "other_ball_action_player",
        "other_ball_action_team",
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


_MIGRATION_3_0_1_MESSAGE_SPORTEC = (
    "silly_kicks.spadl.sportec.convert_to_actions requires explicit per-period "
    "direction info as of silly-kicks 3.0.1. Sportec/DFL bronze events ship "
    "per-period-absolute (teams switch ends after halftime). Pass "
    "`home_team_start_left=True/False` from DFL MatchInformation.xml "
    "(Environment.HomeTeamStartLeftSide), OR pass "
    "`home_attacks_right_per_period={1: bool, 2: bool, ...}` directly. "
    "See ADR-006 erratum + CHANGELOG 3.0.1 for the migration snippet."
)


def _resolve_per_period_flips_sportec(
    *,
    events: pd.DataFrame,
    home_team_start_left: bool | None,
    home_team_start_left_extratime: bool | None,
    home_attacks_right_per_period: Mapping[int, bool] | None,
) -> dict[int, bool]:
    """Resolve the bool-pair-OR-mapping kwargs to a concrete per-period flip dict.

    Mutual exclusion + loud failure on missing-input. Mirrors Gradient Sports
    events-side and tracking-side Sportec API exactly.
    """
    if home_attacks_right_per_period is not None and (
        home_team_start_left is not None or home_team_start_left_extratime is not None
    ):
        raise ValueError(
            "sportec.convert_to_actions: pass home_team_start_left[+_extratime] OR "
            "home_attacks_right_per_period, not both."
        )

    if home_attacks_right_per_period is not None:
        return dict(home_attacks_right_per_period)

    if home_team_start_left is None and home_team_start_left_extratime is not None:
        raise ValueError(
            "sportec.convert_to_actions: home_team_start_left_extratime supplied without "
            "home_team_start_left. ET flag is meaningful only as a refinement of the "
            "regular-time flag."
        )

    if home_team_start_left is None:
        raise ValueError(_MIGRATION_3_0_1_MESSAGE_SPORTEC)

    period_col = "period" if "period" in events.columns else "period_id"
    if period_col in events.columns:
        direction.require_et_direction(
            events[period_col], home_team_start_left_extratime, source="sportec convert_to_actions"
        )

    return direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: str,
    *,
    home_team_start_left: bool | None = None,
    home_team_start_left_extratime: bool | None = None,
    home_attacks_right_per_period: Mapping[int, bool] | None = None,
    preserve_native: list[str] | None = None,
    goalkeeper_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, SportecConversionReport]:
    """Convert normalized Sportec/DFL event DataFrame to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Normalized DFL event data with columns per ``EXPECTED_INPUT_COLUMNS``
        plus any subset of optional qualifier columns.
    home_team_id : str
        Identifier of the home team (used to route per-period coord flips for
        canonical SPADL "all-actions-LTR" orientation).
    home_team_start_left : bool or None, default ``None``
        From DFL ``MatchInformation.xml`` (``Environment.HomeTeamStartLeftSide``
        attribute). When True, the home team starts on the left side of the
        pitch in period 1 (its goal is on the left, so it shoots toward the
        right goal). silly-kicks 3.0.1 introduced this kwarg to fix the
        per-period-absolute direction-of-play bug; pass exactly one of
        ``home_team_start_left`` or ``home_attacks_right_per_period``.
    home_team_start_left_extratime : bool or None, default ``None``
        Same flag for ET periods 3/4 (DFL ``HomeTeamStartLeftSideExtraTime``
        equivalent). Required only when the input contains ``period_id`` 3 or 4.
    home_attacks_right_per_period : Mapping[int, bool] or None, default ``None``
        Escape hatch: pass the per-period flip mapping directly (e.g.
        ``{1: True, 2: False}``). Useful for non-canonical period structures
        (futsal halves, friendly-match formats) or when the caller has
        pre-derived the mapping. Pass exactly one of ``home_team_start_left``
        or ``home_attacks_right_per_period``.
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
    report : SportecConversionReport
        Audit trail with mapped/excluded/unrecognized event-type counts plus the
        F4 BallClaiming and F7 free-kick-crossed diagnostics.

    Examples
    --------
    Convert a Sportec/IDSSE bronze events DataFrame to SPADL -- preferred path::

        from silly_kicks.spadl import sportec

        actions, report = sportec.convert_to_actions(
            events,
            home_team_id="DFL-CLU-XXXXX",
            home_team_start_left=True,    # from DFL MatchInformation.xml
            goalkeeper_ids={"DFL-OBJ-..."},  # optional supplementary GK ids
        )
        # report.mapped_counts gives per-event-type counts; output schema is
        # SPORTEC_SPADL_COLUMNS (KLOPPY_SPADL_COLUMNS + 4 tackle qualifier columns).

    Same conversion via the explicit-mapping escape hatch::

        actions, report = sportec.convert_to_actions(
            events,
            home_team_id="DFL-CLU-XXXXX",
            home_attacks_right_per_period={1: False, 2: True},
        )
    """
    _validate_input_columns(events, EXPECTED_INPUT_COLUMNS, provider="Sportec")
    _validate_preserve_native(events, preserve_native, provider="Sportec", schema=KLOPPY_SPADL_COLUMNS)

    # Normalize DFL XML tag aliases before dispatch (e.g. "CornerKick" → "Corner").
    if events["event_type"].isin(_EVENT_TYPE_ALIASES).any():
        events = events.copy()
        events["event_type"] = events["event_type"].replace(_EVENT_TYPE_ALIASES)

    event_type_counts = Counter(events["event_type"])

    # PR-S23 / silly-kicks 3.0.1: native Sportec/DFL bronze events ship
    # PER_PERIOD_ABSOLUTE (teams switch ends after halftime; verified
    # empirically via lakehouse SK3-MIG probe). The kloppy gateway path
    # (silly_kicks.spadl.kloppy) remains ABSOLUTE_FRAME_HOME_RIGHT --
    # kloppy normalises via Orientation.HOME_AWAY upstream. See ADR-006
    # erratum.
    flips = _resolve_per_period_flips_sportec(
        events=events,
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
        home_attacks_right_per_period=home_attacks_right_per_period,
    )

    # Convention sanity check: declared PER_PERIOD_ABSOLUTE; validator
    # surfaces upstream loader regressions (e.g. caller accidentally feeds
    # absolute-frame data). Strict mode under SILLY_KICKS_ASSERT_INVARIANTS=1.
    _detector_input = events.assign(
        _sk_match_id=events["match_id"] if "match_id" in events.columns else 0,
        _sk_is_shot=(events["event_type"] == "ShotAtGoal"),
    )
    _x_col = "x_source_position" if "x_source_position" in events.columns else "x_position_from_tracking"
    _period_col = "period" if "period" in events.columns else "period_id"
    if _x_col in events.columns and "team" in events.columns and _period_col in events.columns:
        validate_input_convention(
            _detector_input,
            declared=PER_PERIOD_ABSOLUTE,
            match_col="_sk_match_id",
            x_col=_x_col,
            team_col="team",
            period_col=_period_col,
            is_shot_col="_sk_is_shot",
            x_max=105.0,
        )

    raw_actions, claim_counts = _build_raw_actions(
        events, preserve_native, home_team_id=home_team_id, goalkeeper_ids=goalkeeper_ids
    )

    # §25.3: fold a standalone direct-free-kick pair (FreeKick + flagged ShotAtGoal) into one
    # shot_freekick row. Before the chronological sort and the LTR flip (source-orientation coords).
    raw_actions = _pair_direct_free_kicks(raw_actions)

    # Chronological-action_id invariant: sort the raw actions into
    # (game_id, period_id, time_seconds) order at the TOP of the frame -- before
    # any positional (.shift()-based) derivation (_derive_end_coordinates /
    # _add_dribbles) -- so action_id becomes a chronological index and every
    # time-neighbour lookup resolves the true neighbour. Native DFL bronze can
    # ship events out of time order (a genuine ..., 28.806, 26.370 inversion in
    # the IDSSE slice). Empty frame passes through unchanged. See
    # sort_actions_chronologically in .base.
    raw_actions = sort_actions_chronologically(raw_actions)

    if len(raw_actions) > 0:
        actions = _derive_end_coordinates(raw_actions)
        actions = to_spadl_ltr(
            actions,
            input_convention=PER_PERIOD_ABSOLUTE,
            home_team_id=home_team_id,
            home_attacks_right_per_period=flips,
        )
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
    actions = _ensure_base_signal_columns(actions)
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
    # F7 (TF-46 Arm 1): count the free kicks whose subtype fell back to freekick_short because the DFL
    # play_penalty_box qualifier was absent (type_source=="inferred"). Only F7 sets "inferred" on sportec.
    freekick_inferred_count = int((actions["type_source"] == "inferred").sum()) if len(actions) else 0
    report = SportecConversionReport(
        provider="Sportec",
        total_events=sum(event_type_counts.values()),
        total_actions=len(actions),
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        unrecognized_counts=unrecognized_counts,
        ball_claim_counts=dict(claim_counts["ball_claim_counts"]),
        interceptions_added=int(claim_counts["interceptions_added"]),
        keeper_pick_ups_added=int(claim_counts["keeper_pick_ups_added"]),
        ball_claim_unattached=int(claim_counts["ball_claim_unattached"]),
        freekick_inferred_count=freekick_inferred_count,
        penalty_result_method_counts=dict(claim_counts.get("penalty_result_method_counts", {})),
        penalty_retakes=int(claim_counts.get("penalty_retakes", 0)),
        deflection_counts=dict(claim_counts.get("deflection_counts", {})),
        own_goal_attribution_conflicts=int(claim_counts.get("own_goal_attribution_conflicts", 0)),
        penalty_opponent_unresolved=int(claim_counts.get("penalty_opponent_unresolved", 0)),
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
    # int(): np.where yields np.intp, and `pairs` is a dict[int, str]. Casting at the loop
    # boundary keeps every downstream use (dict key, .iloc) plainly typed.
    for idx in (int(_i) for _i in foul_indices):
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


_SUCCESS_EVAL: tuple[str, str] = ("successfullyCompleted", "successful")
# Everything we expect on a DFL pass/set-piece Evaluation; anything else is surfaced by the warn.
_KNOWN_EVAL: frozenset[str] = frozenset(_SUCCESS_EVAL) | {"unsuccessful", ""}


def _extract_play_eval(df: pd.DataFrame) -> np.ndarray:
    """Normalize the optional DFL ``play_evaluation`` column to a clean str array.

    Absent column / NaN / null -> ``""`` so a missing column never mass-fails non-DFL sportec-like
    data. Single source of the extraction both completion sites share (no per-site ``fillna`` drift).
    """
    if "play_evaluation" in df.columns:
        return df["play_evaluation"].fillna("").astype(str).to_numpy()
    return np.full(len(df), "", dtype=object)


def _play_evaluation_is_fail(play_eval: np.ndarray) -> np.ndarray:
    """Success-allowlist completion: non-empty, non-success DFL Evaluation -> fail (kloppy-aligned).

    Empty / absent / null -> not-fail (the conservative success default). Any unseen reason-coded
    failure token (e.g. ``unsuccessfulBecauseOfFoul``) -> fail by construction. Exact camelCase match
    (DFL is consistent camelCase; a case-variant fails+warns by design -- deliberately NOT
    ``.str.lower()`` like sibling qualifiers). Mirrors kloppy ``sportec/deserializer.py`` (the
    reference DFL parser): ``Evaluation in {successfullyCompleted, successful}`` is the success set.
    """
    return (play_eval != "") & ~np.isin(play_eval, _SUCCESS_EVAL)


def _warn_unexpected_play_eval(play_eval: np.ndarray) -> None:
    """Surface any token that is neither a known success nor the known ``unsuccessful`` failure.

    Makes a genuinely-new or benign DFL token visible (so it can be added to ``_SUCCESS_EVAL``)
    instead of silently classified as fail. Called at BOTH completion sites over each site's relevant
    rows -- punt-Play synth parents are excluded from ``is_pass``, so one warn cannot cover both.
    """
    unexpected = set(np.unique(play_eval)) - _KNOWN_EVAL
    if unexpected:
        warnings.warn(
            f"sportec: unexpected play_evaluation token(s) {sorted(unexpected)} treated as fail "
            f"(not in the success allowlist {_SUCCESS_EVAL}); verify against the DFL spec.",
            stacklevel=2,
        )


_WOODWORK_TYPE_MAP: dict[str, str] = {"postleft": "post_left", "postright": "post_right", "crossbar": "crossbar"}


def _route_ball_deflections(events: pd.DataFrame, actions: pd.DataFrame) -> dict[str, int]:
    """Route DFL ``BallDeflection`` annotations by ``deflection_type`` (spec §25.4).

    - **Woodwork** (``postLeft`` / ``postRight`` / ``crossBar``): pair with the same player's preceding
      ``ShotAtGoal`` with ``shot_outcome_type == "woodwork"``, same period, within 3 s. Set
      ``shot_woodwork_type`` on that shot row; where the shot end is the placeholder (end == start), set the
      end to the deflection point. No new row.
    - **Referee:** a documented exclusion, counted ``deflection_referee``.
    - **Any other type, or woodwork with no pairable shot:** counted ``deflection_unpaired`` and warned.

    Never touches ``shot_blocked`` / ``cross_blocked``. Mutates ``actions`` in place; returns the census.
    """
    counts: dict[str, int] = {}
    if "event_type" not in events.columns:
        return counts
    deflections = events[events["event_type"] == "BallDeflection"]
    if deflections.empty:
        return counts
    shot_types = frozenset(
        {
            spadlconfig.actiontype_id["shot"],
            spadlconfig.actiontype_id["shot_freekick"],
            spadlconfig.actiontype_id["shot_penalty"],
        }
    )
    shots_events = events[events["event_type"] == "ShotAtGoal"]
    for _, deflection in deflections.iterrows():
        dtype = str(deflection.get("deflection_type") or "").strip()
        dtype_key = dtype.lower()
        if dtype_key not in _WOODWORK_TYPE_MAP:
            if dtype_key == "referee":
                counts["deflection_referee"] = counts.get("deflection_referee", 0) + 1
            else:
                counts["deflection_unpaired"] = counts.get("deflection_unpaired", 0) + 1
                warnings.warn(
                    f"Sportec: BallDeflection with deflection_type={dtype!r} has no SPADL routing; counted "
                    f"as deflection_unpaired.",
                    stacklevel=2,
                )
            continue

        player = deflection.get("deflection_player")
        period = deflection["period"]
        d_time = float(deflection["timestamp_seconds"])
        cand = shots_events[
            (shots_events["period"] == period)
            & (shots_events["timestamp_seconds"] <= d_time)
            & (shots_events["timestamp_seconds"] >= d_time - 3.0)
        ]
        if "shot_outcome_type" in cand.columns:
            cand = cand[cand["shot_outcome_type"].astype(str).str.lower() == "woodwork"]
        if "shot_player" in cand.columns:
            cand = cand[cand["shot_player"].map(canonical_id) == canonical_id(player)]
        if cand.empty:
            counts["deflection_unpaired"] = counts.get("deflection_unpaired", 0) + 1
            warnings.warn(
                f"Sportec: woodwork BallDeflection (deflection_type={dtype!r}) has no pairable ShotAtGoal; "
                f"counted as deflection_unpaired.",
                stacklevel=2,
            )
            continue
        shot_event = cand.sort_values("timestamp_seconds").iloc[-1]
        target = actions[actions["original_event_id"].map(canonical_id) == canonical_id(shot_event["event_id"])]
        target = target[target["type_id"].isin(shot_types)]
        if target.empty:
            counts["deflection_unpaired"] = counts.get("deflection_unpaired", 0) + 1
            warnings.warn(
                f"Sportec: woodwork BallDeflection (deflection_type={dtype!r}) paired to a ShotAtGoal event "
                f"with no surviving SPADL shot row; counted as deflection_unpaired.",
                stacklevel=2,
            )
            continue
        j = target.index[0]
        actions.loc[j, "shot_woodwork_type"] = _WOODWORK_TYPE_MAP[dtype_key]
        end_is_placeholder = actions.loc[j, "end_x"] == actions.loc[j, "start_x"] and (
            actions.loc[j, "end_y"] == actions.loc[j, "start_y"]
        )
        if end_is_placeholder:
            actions.loc[j, "end_x"] = float(deflection["x"])
            actions.loc[j, "end_y"] = float(deflection["y"])
        counts["woodwork_paired"] = counts.get("woodwork_paired", 0) + 1
    return counts


def _pair_direct_free_kicks(actions: pd.DataFrame) -> pd.DataFrame:
    """Fold a standalone direct-free-kick pair into one ``shot_freekick`` row (spec §25.3).

    A ``FreeKick`` row followed by the same player's ``ShotAtGoal`` with ``shot_after_free_kick == "true"``
    within 5 s (same period) is one direct free kick recorded as two DFL events. The shot already types
    ``shot_freekick`` (the ``after_fk`` dispatch branch); keep it, move the FreeKick's start onto it, and
    drop the FreeKick row. Mirrors the Metrica in-converter pairing. Runs on source-orientation coords,
    before the LTR flip, so both rows reflect together. The case does not occur in IDSSE (0 of 164 shots
    carry the flag).
    """
    if len(actions) == 0:
        return actions
    fk_types = frozenset({spadlconfig.actiontype_id["freekick_short"], spadlconfig.actiontype_id["freekick_crossed"]})
    shot_freekick = spadlconfig.actiontype_id["shot_freekick"]
    out = actions.reset_index(drop=True)
    drop_rows: list[int] = []
    used_shots: set[int] = set()
    for i in out.index[out["type_id"].isin(fk_types)]:
        fk = out.loc[i]
        same_player = ids_match(out["player_id"], fk["player_id"])
        cand_mask = (
            (out["type_id"] == shot_freekick)
            & (out["period_id"] == fk["period_id"])
            & same_player
            & (out["time_seconds"] >= fk["time_seconds"])
            & (out["time_seconds"] <= fk["time_seconds"] + 5.0)
        )
        cand = [j for j in out.index[cand_mask] if j not in used_shots]
        if not cand:
            continue
        j = cand[0]
        out.loc[j, "start_x"] = fk["start_x"]
        out.loc[j, "start_y"] = fk["start_y"]
        used_shots.add(j)
        drop_rows.append(int(i))
    if drop_rows:
        out = out.drop(index=drop_rows).reset_index(drop=True)
    return out


def _parse_final_score(events: pd.DataFrame) -> tuple[int, int] | None:
    """Return (home_goals, away_goals) from the last period's FinalWhistle ``whistle_final_result``.

    The DFL ``whistle_final_result`` is carried on the ``FinalWhistle`` event, one per period, in
    ``home:away`` order. The full-time score is the value on the match's last period. Returns None when the
    column is absent, empty or unparseable.
    """
    if "whistle_final_result" not in events.columns:
        return None
    carriers = events[events["whistle_final_result"].notna()]
    if carriers.empty:
        return None
    last = carriers.sort_values(["period", "timestamp_seconds"]).iloc[-1]["whistle_final_result"]
    try:
        home_str, away_str = str(last).split(":")
        return int(home_str), int(away_str)
    except (ValueError, AttributeError):
        return None


def _eliminate(r: int, unresolved: list[int]) -> dict[int, tuple[int, str]]:
    """Resolve the §25.2 step-3b elimination for one team (CL-14).

    ``r`` = reconciled penalty goals minus the successes already resolved; ``unresolved`` = the row
    positions of the team's still-unresolved new-format penalties. Resolves all unresolved to ``fail`` when
    r == 0, all to ``success`` when r == len(unresolved), and nothing otherwise. The R == U branch is
    unreachable at match level (CL-14), so it is exercised only through this pure helper.
    """
    success = spadlconfig.result_id["success"]
    fail = spadlconfig.result_id["fail"]
    if not unresolved:
        return {}
    if r == 0:
        return {i: (fail, "score_elimination") for i in unresolved}
    if r == len(unresolved):
        return {i: (success, "score_elimination") for i in unresolved}
    return {}


def _match_team_pair(events: pd.DataFrame) -> list[object]:
    """Return the two real match team role labels.

    Taken from the ``KickOff`` events (both teams kick off at least at their period starts); the generic
    ``team`` column also carries non-team actors (e.g. ``"unknown"`` on every IDSSE match), so it cannot be
    used directly (TF-46 Arm 1 Part 2, finding B-01 / C-1). Fallback: the two most frequent non-``unknown``
    labels.
    """
    if "event_type" in events.columns:
        ko_teams = [t for t in events.loc[events["event_type"] == "KickOff", "team"].dropna().unique()]
        if len(ko_teams) == 2:
            return ko_teams
    labels = events["team"].dropna()
    labels = labels[labels.astype(str).str.lower() != "unknown"]
    return list(labels.value_counts().index[:2])


def _opponent_label(team: object, match_teams: list[object]) -> object | None:
    """Return the other of the two match team labels, or None when it cannot be resolved."""
    others = [t for t in match_teams if str(t) != str(team)]
    return others[0] if len(others) == 1 else None


def _clu_to_role(rows: pd.DataFrame) -> dict[str, object]:
    """Map each team's DFL-CLU id (``team_id_native``) to its role label (``team``), from the bronze rows."""
    mapping: dict[str, object] = {}
    if "team_id_native" not in rows.columns:
        return mapping
    for clu, role in zip(rows["team_id_native"], rows["team"], strict=False):
        if clu is not None and not pd.isna(clu) and role is not None and not pd.isna(role):
            mapping[str(canonical_id(clu))] = role
    return mapping


# The DFL event types that map to an on-ball SPADL action (play resuming). Fouls are stoppages, cards /
# substitutions / VAR actions / FinalWhistle / annotations are not on-ball, and Penalty / KickOff are
# handled separately -- so none of those count as "continued play" after a penalty (spec §25.2 step 3m).
_ON_BALL_EVENT_TYPES: frozenset[str] = frozenset(
    {"Play", "ShotAtGoal", "FreeKick", "Corner", "ThrowIn", "GoalKick", "TacklingGame", "OtherBallAction"}
)


def _first_restart_signal(events: pd.DataFrame, opp: object, i: int, rows: pd.DataFrame) -> str:
    """§25.2 steps 3 / 3m: the first restart signal after a penalty, in its period.

    ``"kickoff"`` when an opponent ``KickOff`` comes first (the penalty scored; a period-start kick-off
    precedes the penalty, so ``t > pen_t`` already excludes it). ``"continued_play"`` when an on-ball event
    comes first (play resumed without a goal kick-off, so the penalty did not score). ``"none"`` when neither
    is observed, so the penalty falls to step 3b / step 4.

    Two premise gates keep the signal to what the feed actually observes (C-16):

    - **Kick-off premise.** If the penalty's period codes NO ``KickOff`` event at all, an absent opponent
      kick-off proves nothing, so return ``"none"`` whether or not the opponent resolves. On IDSSE every
      period has a kick-off, so real output is unchanged.
    - **Unresolvable opponent.** When ``opp is None`` a ``KickOff`` cannot be attributed. An on-ball event
      before ANY team's kick-off is still an observed ``"continued_play"`` (no kick-off of any team means no
      opponent kick-off). A kick-off first -> ``"none"`` (attributing it needs the opponent; a non-team-label
      kick-off is never assumed to be the opponent's).
    """
    period_events = events[events["event_type"].notna() & (events["period"] == int(rows["period"].iloc[i]))]
    if not bool((period_events["event_type"] == "KickOff").any()):
        return "none"  # kick-off premise: the feed codes no kick-off this period -> no step-3/3m evidence
    pen_t = float(rows["timestamp_seconds"].iloc[i])
    after = period_events[period_events["timestamp_seconds"] > pen_t]
    if after.empty:
        return "none"
    on_ball = after[after["event_type"].isin(_ON_BALL_EVENT_TYPES)]
    t_on_ball = float(on_ball["timestamp_seconds"].min()) if not on_ball.empty else float("inf")
    if opp is None:
        any_kickoff = after[after["event_type"] == "KickOff"]
        t_any_kickoff = float(any_kickoff["timestamp_seconds"].min()) if not any_kickoff.empty else float("inf")
        if t_on_ball == float("inf"):
            return "none"
        # an on-ball before any team's kick-off is observed continued play; a kick-off first is not evidence.
        return "continued_play" if t_on_ball < t_any_kickoff else "none"
    opp_kickoff = after[(after["event_type"] == "KickOff") & (after["team"].astype(str) == str(opp))]
    t_kickoff = float(opp_kickoff["timestamp_seconds"].min()) if not opp_kickoff.empty else float("inf")
    if t_kickoff == float("inf") and t_on_ball == float("inf"):
        return "none"
    return "kickoff" if t_kickoff <= t_on_ball else "continued_play"


def _resolve_penalty_step4(
    unresolved: list[int],
    reconciled: int | None,
    resolved: dict[int, tuple[int, str, str]],
    repair_kind: np.ndarray,
    *,
    team: object,
    success: int,
    fail: int,
) -> None:
    """§25.2 step 4 (inferred): resolve the still-unresolved penalties, mutating ``resolved`` + ``repair_kind``.

    With a final score, R' = clamp(R, 0, U) keeps the team total equal to the final score: the earliest R'
    penalties (chronological) are success, the rest fail. An R outside [0, U] is a match-level
    inconsistency -> ``repair_kind="penalty_score_inconsistent"`` and a warn (Guard 1 then fails by design).
    With no final score (malformed DFL input; ``whistle_final_result`` is required) -> all fail.
    """
    count_unresolved = len(unresolved)
    if reconciled is None:
        for i in unresolved:
            resolved[i] = (fail, "inferred", "unresolved")
            repair_kind[i] = "penalty_outcome_unresolved"
        return
    remaining = reconciled - sum(1 for v in resolved.values() if v[0] == success)
    r_prime = max(0, min(remaining, count_unresolved))
    inconsistent = not (0 <= remaining <= count_unresolved)
    repair = "penalty_score_inconsistent" if inconsistent else "penalty_outcome_unresolved"
    for rank, i in enumerate(unresolved):  # unresolved is chronological (pen_idx order)
        resolved[i] = (success if rank < r_prime else fail, "inferred", "unresolved")
        repair_kind[i] = repair
    if inconsistent:
        warnings.warn(
            f"sportec: penalty score inconsistent for team {team!r} "
            f"(reconciled remainder {remaining} outside [0, {count_unresolved}]); "
            f"the penalties are marked penalty_score_inconsistent and Guard 1 will fail.",
            stacklevel=2,
        )


def _penalty_results(
    events: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    is_penalty_survivor: np.ndarray,
    home_team_id: str | None,
    et: np.ndarray,
    shot_outcome: np.ndarray,
    result_ids: np.ndarray,
    result_source: np.ndarray,
    penalty_method: np.ndarray,
    repair_kind: np.ndarray,
    teams: np.ndarray,
    owngoal_credit: np.ndarray,
    match_teams: list[object],
) -> int:
    """Resolve new-format DFL penalty results per team (§25.2 steps 2-4), mutating the passed arrays.

    The counting set is the team's new-format ``Penalty`` survivors only (CL-14); legacy ``ShotAtGoal``
    penalties are resolved at step 1 and their goals already sit inside "ShotAtGoal goals". ``teams`` is the
    field-resolved team per row (own-goal rows carry the conceder, decision 3); ``owngoal_credit`` is the
    field-authoritative CREDITED (benefiting) team per own-goal row, or ``None`` when the credit is unknown
    (C-17 option B); ``match_teams`` is the two real match labels (B-01: the generic ``team`` column also
    carries ``unknown``).

    Returns the number of teams whose opponent was unresolvable (``penalty_opponent_unresolved``; 0 on a
    well-formed two-team feed) -- a visibility counter for the C-16 degenerate-frame signal.
    """
    success = spadlconfig.result_id["success"]
    fail = spadlconfig.result_id["fail"]
    final = _parse_final_score(events)
    is_shot = et == "ShotAtGoal"
    is_goal = is_shot & np.isin(shot_outcome, ["goal", "successful"])
    # Own goals are counted by RESULT, so a top-level DFL OwnGoal event (bad_touch / owngoal, decision 3) and
    # the defensive shot-outcome path both count. teams[i] is the team that scored into its own net, so the
    # own goal is credited to that team's opponent (the §25.2 "opponent own goals" term).
    owngoal_result = spadlconfig.result_id["owngoal"]
    is_owngoal = result_ids == owngoal_result

    survivor_idx = np.where(is_penalty_survivor)[0]
    # Fail loud (ADR-051 "unresolved is a value, never a silent default"; ADR-055 "refuse at the edge").
    # C-17 (option B): refuse only when an own goal's CREDIT is unknown -- no TeamOwnGoalCountsFor mapping and
    # no resolved conceder (``owngoal_credit`` is None on that row). The field is provider truth (C-3: the
    # credit is resolved, the goal is never lost); an unknown credit cannot be reconciled for either team, so
    # refuse. This closes both the unresolved-opponent path and the resolved-opponent unknown-conceder path
    # (B-12) with one condition. A known credit never fires, whether or not the opponent resolves.
    if bool((is_owngoal & pd.isna(owngoal_credit)).any()):
        raise ValueError(
            f"cannot reconcile penalties: an own goal has an unknown credit (no TeamOwnGoalCountsFor mapping "
            f"and no resolved conceder; match_teams={match_teams!r}); every own goal must name a credited team"
        )
    credit_str = np.array([str(c) for c in owngoal_credit], dtype=object)
    penalty_opponent_unresolved = 0
    for team in pd.unique(teams[survivor_idx]):
        pen_idx = sorted(
            (int(i) for i in survivor_idx if str(teams[i]) == str(team)),
            key=lambda i: float(rows["timestamp_seconds"].iloc[i]),
        )
        count = len(pen_idx)
        resolved: dict[int, tuple[int, str, str]] = {}

        # The opponent is needed only for the step-3 kick-off signal; step 2 reads own-goal credit directly.
        # When the opponent is unresolvable, steps 3 / 3m emit no opponent-dependent result (see
        # `_first_restart_signal`) and the team's penalties fall to step 4. Make the degenerate frame visible
        # (C-16): warn and count it. On a well-formed two-team feed this never fires.
        opp = _opponent_label(team, match_teams)
        if opp is None:
            penalty_opponent_unresolved += 1
            warnings.warn(
                f"sportec: the opponent of {team!r} is unresolvable (match_teams={match_teams!r}); penalty "
                f"steps 3/3m emit no opponent-dependent result for this team. A degenerate feed -- real DFL "
                f"codes two kick-off teams.",
                stacklevel=2,
            )

        reconciled: int | None = None
        if final is not None:
            home_goals, away_goals = final
            is_home = home_team_id is not None and str(team) == str(home_team_id)
            team_final = home_goals if is_home else away_goals
            # Step-2 term: own goals CREDITED to this team (field-authoritative via ``owngoal_credit``), NOT
            # "own goals conceded by opp" -- so it holds even when the opponent is unresolvable (C-17 B).
            team_owngoals = int((is_owngoal & (credit_str == str(team))).sum())
            team_shotgoals = int((is_goal & (teams == team)).sum())
            reconciled = team_final - team_shotgoals - team_owngoals
            # Step 2: all-or-nothing on the team's penalty count.
            if reconciled == 0:
                resolved = {i: (fail, "native", "score_reconciliation") for i in pen_idx}
            elif reconciled == count:
                resolved = {i: (success, "native", "score_reconciliation") for i in pen_idx}

        if len(resolved) < count:
            # Steps 3 + 3m: the first restart signal after the penalty decides. An opponent kick-off before
            # any on-ball event -> the penalty scored (kickoff_order). An on-ball event before any kick-off
            # -> play continued, so the penalty did not score (continued_play). Both are native. When opp is
            # None, `_first_restart_signal` returns "continued_play" only from an observed on-ball before any
            # team's kick-off, and "none" (no native result) when a kick-off comes first, since attributing it
            # needs the opponent (C-16). A period with no kick-off at all also yields "none" (an absent
            # kick-off proves nothing). Penalties left unresolved fall to step 3b / step 4.
            for i in pen_idx:
                if i in resolved:
                    continue
                signal = _first_restart_signal(events, opp, i, rows)
                if signal == "kickoff":
                    resolved[i] = (success, "native", "kickoff_order")
                elif signal == "continued_play":
                    resolved[i] = (fail, "native", "continued_play")
            # Step 3b: native elimination to a fixed point (one pass suffices for one team).
            if reconciled is not None:
                successes = sum(1 for v in resolved.values() if v[0] == success)
                remaining = reconciled - successes
                unresolved = [i for i in pen_idx if i not in resolved]
                for i, (rid, method) in _eliminate(remaining, unresolved).items():
                    resolved[i] = (rid, "native", method)
            # Step 4: still unresolved -> inferred. With a final score, split by R' = clamp(R, 0, U) so the
            # team total still equals the final score (Part-1 missing_goal precedent); the earliest R'
            # penalties are success. An R outside [0, U] is a match-level inconsistency: flag it and warn.
            unresolved = [i for i in pen_idx if i not in resolved]
            if unresolved:
                _resolve_penalty_step4(
                    unresolved, reconciled, resolved, repair_kind, team=team, success=success, fail=fail
                )

        for i, (rid, source, method) in resolved.items():
            result_ids[i] = rid
            result_source[i] = source
            penalty_method[i] = method

    return penalty_opponent_unresolved


def _build_raw_actions(
    events: pd.DataFrame,
    preserve_native: list[str] | None,
    *,
    home_team_id: str | None = None,
    goalkeeper_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build raw SPADL actions DataFrame from DFL event rows.

    For each row, dispatches on ``event_type`` to determine SPADL
    type/result/bodypart. Drops rows whose ``event_type`` is in
    ``_EXCLUDED_EVENT_TYPES`` or not in ``_MAPPED_EVENT_TYPES``.
    Direction-of-play flipping happens later in ``convert_to_actions``;
    this function emits source-orientation coords.

    Returns the raw actions frame and the F4 ``BallClaiming`` audit counts
    (``_ball_claim_report_template`` shape).
    """
    mask = events["event_type"].isin(_MAPPED_EVENT_TYPES)
    rows = events.loc[mask].copy().reset_index(drop=True)
    n = len(rows)

    if n == 0:
        return _empty_raw_actions(preserve_native, events), _ball_claim_report_template()

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
    # DFL bronze emits play_flat_cross as the native string "true"/"false" (NOT a bool).
    # .astype(bool) maps any non-empty string — including "false" — to True, which would
    # mislabel ~99% of passes as crosses. Parse the string explicitly, matching the
    # shot_after_free_kick / *_defensive_clearance siblings below. str(True).lower()=="true"
    # so this also handles a native-bool column correctly.
    is_cross_by_flag = is_pass & _opt("play_flat_cross", "").fillna("").astype(str).str.lower().eq("true").to_numpy()
    is_cross = is_cross_by_height | is_cross_by_flag
    is_pass_plain = is_pass & ~is_cross
    type_ids[is_pass_plain] = spadlconfig.actiontype_id["pass"]
    type_ids[is_cross] = spadlconfig.actiontype_id["cross"]
    result_ids[is_pass] = spadlconfig.result_id["success"]

    # Pass head bodypart
    is_head = is_pass & _opt("play_height", None).eq("head").to_numpy()
    bodypart_ids[is_head] = spadlconfig.bodypart_id["head"]

    # --- Set pieces ---
    # F7 (TF-46 Arm 1): the free-kick subtype reads the DFL ``play_penalty_box`` qualifier, NOT the old
    # ``freekick_execution_mode`` crossed mapping whose {cross,long,longBall,highPass} values never occur in
    # DFL (IDSSE emits only direct/indirect), so every DFL free kick was silently mistyped freekick_short.
    # ``play_penalty_box == "true"`` -> freekick_crossed; "false" -> freekick_short; missing -> freekick_short
    # with type_source="inferred" (counted in the report). The subtype is the provider flag, not a derived
    # box-delivery judgement (ADR-127 fix class). See the TF-46 Arm 1 corrections spec section 7.3.
    is_freekick = et == "FreeKick"
    pen_box = _opt("play_penalty_box", "").fillna("").astype(str).str.lower().to_numpy()
    fk_pen_box_present = is_freekick & np.isin(pen_box, ["true", "false"])
    fk_pen_box_missing = is_freekick & ~fk_pen_box_present
    is_fk_crossed = is_freekick & (pen_box == "true")
    type_ids[is_freekick & ~is_fk_crossed] = spadlconfig.actiontype_id["freekick_short"]
    type_ids[is_fk_crossed] = spadlconfig.actiontype_id["freekick_crossed"]
    result_ids[is_freekick] = spadlconfig.result_id["success"]
    # type_source: native where the flag drives the subtype, inferred where the flag is absent (stopgap
    # freekick_short). Base-schema column (decision D); NA elsewhere, filled by _ensure_base_signal_columns.
    type_source = np.full(n, None, dtype=object)
    type_source[fk_pen_box_present] = "native"
    type_source[fk_pen_box_missing] = "inferred"

    is_corner = et == "Corner"
    corner_target = _opt("corner_target_area", "").fillna("").to_numpy()
    corner_placing = _opt("corner_placing", "").fillna("").to_numpy()
    corner_height = _opt("play_height", "").fillna("").to_numpy()
    # A corner is CROSSED when delivered aerially into the box, else corner_short. The primary
    # discriminator is the DFL ``play_height`` qualifier on the corner's Play: on CornerKick events it
    # takes {high, flat} (distinct from the {cross, head} vocabulary it carries on open-play passes).
    # Measured across the 7 IDSSE matches (57 corners): play_height=="high" ⟺ corner_target_area in
    # {center, nearPost, farPost, centerOutsideBox, outOfPlay} (53 crossed); play_height=="flat" ⟺
    # corner_target_area=="short" (4 short) -- a clean partition. The older
    # {box, penaltyBox, sixYardBox}/{aerial, high} tokens matched NO real DFL value (they came from a
    # self-referential unit fixture), so every DFL corner was silently mistyped corner_short; they are
    # kept in the union only for backward-compatibility with any pre-normalized bronze input.
    is_corner_crossed = is_corner & (
        (corner_height == "high")
        | np.isin(corner_target, ["box", "penaltyBox", "sixYardBox"])
        | np.isin(corner_placing, ["aerial", "high"])
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

    # Pass-class + set-piece completion from the native DFL `play_evaluation` (carried on Play AND on
    # GoalKick/FreeKick/Corner/ThrowIn via their nested Play). Success-ALLOWLIST (kloppy-aligned, 4.21.3):
    # `fail` iff the Evaluation is non-empty AND not in {successfullyCompleted, successful} -- so any
    # unseen reason-coded failure token (e.g. `unsuccessfulBecauseOfFoul`) fails by construction, while
    # empty/absent/NULL stays success (conservative; never mass-fails a missing column). Byte-identical
    # on observed DFL data (the only non-success token across the 7 IDSSE matches is `unsuccessful`).
    play_eval = _extract_play_eval(rows)
    is_pass_or_setpiece = is_pass | is_freekick | is_corner | is_throwin | is_goalkick
    result_ids[is_pass_or_setpiece & _play_evaluation_is_fail(play_eval)] = spadlconfig.result_id["fail"]
    _warn_unexpected_play_eval(play_eval[is_pass_or_setpiece])

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
    # DFL/IDSSE uses "successful" for goals; legacy format uses "goal".
    is_goal = is_shot & np.isin(shot_outcome, ["goal", "successful"])
    is_owngoal = is_shot & (shot_outcome == "ownGoal")
    result_ids[is_shot] = spadlconfig.result_id["fail"]
    result_ids[is_goal] = spadlconfig.result_id["success"]
    # Owngoals coded as a ShotAtGoal outcome: emit as bad_touch + result=owngoal. DFL does NOT use this path
    # (it codes an own goal as a top-level OwnGoal event, handled below); kept defensive for other feeds.
    type_ids[is_owngoal] = spadlconfig.actiontype_id["bad_touch"]
    result_ids[is_owngoal] = spadlconfig.result_id["owngoal"]

    # --- Own goal (top-level DFL OwnGoal event; TF-46 Arm 1 Part 2 decision 3, DL ruling 2026-10-21) ---
    # DFL codes an own goal as its own top-level event, not a ShotAtGoal outcome. Map it to an ADR-018
    # bad_touch / owngoal; the body part comes from own_goal_type_of_shot. The CREDIT is field-authoritative:
    # own_goal_team_own_goal_counts_for (a DFL-CLU id) names the team the goal counts FOR, resolved to a role
    # label via team_id_native <-> team. The owngoal ROW's team_id is the conceder (the opponent of the
    # credited team); the field RESOLVES it when the native role is unknown, and a known native role that
    # disagrees is a counted conflict (own_goal_attribution_conflicts). §25.2 step 2 reads these rows by the
    # field-resolved team. The 7 IDSSE matches contain 0 OwnGoal events, so IDSSE totals are unchanged.
    is_own_goal_event = et == "OwnGoal"
    type_ids[is_own_goal_event] = spadlconfig.actiontype_id["bad_touch"]
    result_ids[is_own_goal_event] = spadlconfig.result_id["owngoal"]
    _og_shot_type = _opt("own_goal_type_of_shot", "").fillna("").astype(str).str.lower()
    is_head_own_goal = is_own_goal_event & _og_shot_type.str.contains("head").to_numpy()
    bodypart_ids[is_head_own_goal] = spadlconfig.bodypart_id["head"]

    match_teams = _match_team_pair(events)
    match_team_strs = {str(t) for t in match_teams}
    team_id_arr = rows["team"].astype("object").to_numpy().copy()
    # owngoal_credit[i] = the CREDITED (benefiting) team per own-goal row, or None when the credit is unknown
    # (C-17 option B). Built for EVERY row `_penalty_results` counts as an own goal BY RESULT -- the top-level
    # DFL OwnGoal event AND the defensive ShotAtGoal "ownGoal" path above -- so the build mask and the consume
    # mask are the same set (C1-IMPL-18; a mismatch spuriously refused a reconcilable shot-outcome own goal).
    # Resolution order: (1) TeamOwnGoalCountsFor -> role (provider truth, wins; OwnGoal events only);
    # (2) else the opponent of the conceder, when the conceder (native team) is a resolved match team;
    # (3) else unknown.
    owngoal_credit = np.full(n, None, dtype=object)
    own_goal_attribution_conflicts = 0
    is_owngoal_result = result_ids == spadlconfig.result_id["owngoal"]
    if is_owngoal_result.any():
        clu_role = _clu_to_role(rows) if is_own_goal_event.any() else {}
        counts_for = _opt("own_goal_team_own_goal_counts_for", None).astype("object").to_numpy()
        for i in np.where(is_owngoal_result)[0]:
            cf = counts_for[i] if is_own_goal_event[i] else None  # the field is an OwnGoal-event qualifier
            credited_role = clu_role.get(str(canonical_id(cf))) if (cf is not None and not pd.isna(cf)) else None
            if credited_role is not None:
                # Rule 1: the field credits the benefiting team directly, even when the opponent / conceder
                # cannot otherwise be resolved (so the goal is never lost; C-3).
                owngoal_credit[i] = credited_role
                scorer_role = _opponent_label(credited_role, match_teams)
                if scorer_role is not None:
                    native = team_id_arr[i]
                    native_known = native is not None and not pd.isna(native) and str(native).lower() != "unknown"
                    if native_known and str(native) != str(scorer_role):
                        own_goal_attribution_conflicts += 1
                    team_id_arr[i] = scorer_role  # field-resolved conceder (authoritative)
                continue
            # Rule 2 / 3: no field (every shot-outcome own goal, and a field-absent OwnGoal event) -> opponent
            # of the conceder when the conceder is a resolved match team, else unknown (None) ->
            # _penalty_results refuses if penalties exist.
            conceder = team_id_arr[i]
            conceder_resolved = conceder is not None and not pd.isna(conceder) and str(conceder) in match_team_strs
            owngoal_credit[i] = _opponent_label(conceder, match_teams) if conceder_resolved else None
        if own_goal_attribution_conflicts:
            warnings.warn(
                f"sportec: {own_goal_attribution_conflicts} OwnGoal event(s) whose TeamOwnGoalCountsFor "
                f"disagrees with the native team; the field is authoritative.",
                stacklevel=2,
            )

    # --- Penalty (TF-46 Arm 1 Part 2 §25) ---
    # Base signal / provenance arrays carried onto the output (mirrors the type_source None-default
    # pattern above). result_source stays None off penalty rows -> _ensure_base_signal_columns leaves it
    # <NA>; penalty_result_method / repair_kind are declared Sportec columns (<NA> off penalty rows).
    result_source = np.full(n, None, dtype=object)
    penalty_method = np.full(n, None, dtype=object)
    repair_kind = np.full(n, None, dtype=object)

    # Step 1 (legacy): a ShotAtGoal with penalty qualifiers (is_shot_penalty) keeps its shot_outcome result
    # and is marked native/shot_outcome. It never enters the §25.2 reconciliation counting set (CL-14).
    result_source[is_shot_penalty] = "native"
    penalty_method[is_shot_penalty] = "shot_outcome"

    # New-format DFL Penalty event -> shot_penalty. A retaken attempt (penalty_retaken_penalty == "true")
    # is a superseded kick: drop it (non_action) and count it; only the final attempt is the shot (CL-12).
    is_penalty_event = et == "Penalty"
    type_ids[is_penalty_event] = spadlconfig.actiontype_id["shot_penalty"]
    pen_retaken = _opt("penalty_retaken_penalty", "").fillna("").astype(str).str.lower().eq("true").to_numpy()
    is_retaken_penalty = is_penalty_event & pen_retaken
    type_ids[is_retaken_penalty] = spadlconfig.actiontype_id["non_action"]
    penalty_retakes = int(is_retaken_penalty.sum())
    is_penalty_survivor = is_penalty_event & ~is_retaken_penalty
    penalty_opponent_unresolved = 0
    if is_penalty_survivor.any():
        penalty_opponent_unresolved = _penalty_results(
            events,
            rows,
            is_penalty_survivor=is_penalty_survivor,
            home_team_id=home_team_id,
            et=et,
            shot_outcome=shot_outcome,
            result_ids=result_ids,
            result_source=result_source,
            penalty_method=penalty_method,
            repair_kind=repair_kind,
            teams=team_id_arr,
            owngoal_credit=owngoal_credit,
            match_teams=match_teams,
        )
    # player_id for a new-format Penalty event: the native player_id is empty, so the taker is sourced
    # from shot_player, else penalty_prospective_taker, else <NA> (spec §25.2; approved caller-id-absent
    # deviation from ADR-001). team_id and start stay the native values (the Penalty event carries the
    # home/away role label and the rule-snapped spot in x/y).
    player_id_arr = rows["player_id"].astype("object").to_numpy().copy()
    if is_penalty_event.any():
        _taker = _opt("shot_player", None).astype("object")
        _ppt = _opt("penalty_prospective_taker", None).astype("object")
        _nonempty = _taker.notna() & (_taker.astype(str).str.len() > 0)
        _taker = _taker.where(_nonempty, _ppt)
        _nonempty = _taker.notna() & (_taker.astype(str).str.len() > 0)
        _taker = _taker.where(_nonempty)  # cond False -> NaN (no taker); object column stays object
        _taker_vals = _taker.to_numpy()
        player_id_arr[is_penalty_event] = _taker_vals[is_penalty_event]

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

    # --- OtherBallAction (DFL XML tag) ---
    # DefensiveClearance=true → clearance; otherwise → non_action. DFL OtherBallAction
    # has NO sub-type qualifier beyond DefensiveClearance (BallPossessionPhase is a
    # possession-sequence id, not a class) and no published definition (IDSSE Scientific
    # Data paper), so a non-clearance one has no concrete SPADL action type. kloppy's
    # Sportec deserializer likewise keeps it a GENERIC (untyped) event. non_action is the
    # honest mapping -- NOT a placeholder "preserved for VAEP context" (the row is dropped).
    # Column name varies by caller: "otherball_defensive_clearance" (recognized
    # qualifier list) or "other_ball_action_defensive_clearance" (lakehouse).
    is_other_ball = et == "OtherBallAction"
    _dc_short = _opt("otherball_defensive_clearance", "").fillna("").astype(str).str.lower().eq("true").to_numpy()
    _dc_long = (
        _opt("other_ball_action_defensive_clearance", "").fillna("").astype(str).str.lower().eq("true").to_numpy()
    )
    is_other_ball_clearance = is_other_ball & (_dc_short | _dc_long)
    type_ids[is_other_ball_clearance] = spadlconfig.actiontype_id["clearance"]
    result_ids[is_other_ball_clearance] = spadlconfig.result_id["success"]
    type_ids[is_other_ball & ~is_other_ball_clearance] = spadlconfig.actiontype_id["non_action"]

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

    # --- Block detection (TF-51 prereq) ---
    # shot_blocked: True on a shot with shot_outcome=="blocked", EXCEPT an own-team
    # deflection (shot_outcome_blocked_by_own_team truthy), which is not an opponent
    # block. cross_blocked: pd.NA -- DFL has no blocked-cross field.
    _own_team_block = (
        _opt("shot_outcome_blocked_by_own_team", "").fillna("").astype(str).str.lower().eq("true").to_numpy()
    )
    _shot_blocked_mask = (shot_outcome == "blocked") & ~_own_team_block

    # action_provenance (F4 / decision B2): "native" for a row lifted 1:1 from a DFL event, "derived" for
    # a converter-synthesized row. The throwOut/punt keeper_pick_up (and its distribution second half, added
    # by _synthesize_gk_distribution_actions) and the supplementary-path keeper_pick_up are synthesized ->
    # "derived"; the F4 BallClaiming interception / keeper_pick_up rows are marked "derived" by the F4 helper.
    action_provenance = np.full(n, "native", dtype=object)
    action_provenance[play_gk_distribution | play_gk_supplementary] = "derived"

    # Assemble main actions DataFrame (1:1 with rows).
    actions = pd.DataFrame(
        {
            "game_id": rows["match_id"].astype("object"),
            "original_event_id": rows["event_id"].astype("object"),
            "period_id": rows["period"].astype(np.int64),
            "time_seconds": rows["timestamp_seconds"].astype(np.float64),
            "team_id": team_id_arr,
            "player_id": player_id_arr,
            "start_x": rows["x"].astype(np.float64),
            "start_y": rows["y"].astype(np.float64),
            "end_x": rows["x"].astype(np.float64),
            "end_y": rows["y"].astype(np.float64),
            "type_id": type_ids,
            "result_id": result_ids,
            "bodypart_id": bodypart_ids,
            "shot_blocked": _blocked_flag(n, applicable=is_shot, blocked=_shot_blocked_mask),
            "cross_blocked": _blocked_flag(n),
            "tackle_winner_player_id": tackle_winner_player_arr,
            "tackle_winner_team_id": tackle_winner_team_arr,
            "tackle_loser_player_id": tackle_loser_player_arr,
            "tackle_loser_team_id": tackle_loser_team_arr,
            "action_provenance": action_provenance,
            "ball_claim_type": np.full(n, None, dtype=object),
            "type_source": type_source,
            "result_source": result_source,
            "penalty_result_method": penalty_method,
            "repair_kind": repair_kind,
            "shot_woodwork_type": np.full(n, None, dtype=object),
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

    # F4 (TF-46 Arm 1): fold DFL BallClaiming annotations in -- set ball_claim_type on the partner action
    # and emit derived interception / keeper_pick_up rows. Runs on the main 1:1 frame BEFORE the non_action
    # drop so partner matching sees (and explicitly excludes) non_action rows. See _synthesize_ball_claiming_actions.
    claim_arr, claim_derived, claim_counts = _synthesize_ball_claiming_actions(events, actions, preserve_native)
    actions["ball_claim_type"] = claim_arr

    # §25: penalty result-method census (legacy shot_outcome + new-format reconciled) and dropped retakes.
    claim_counts["penalty_result_method_counts"] = dict(Counter(m for m in penalty_method if m is not None))
    claim_counts["penalty_retakes"] = penalty_retakes
    claim_counts["own_goal_attribution_conflicts"] = own_goal_attribution_conflicts
    claim_counts["penalty_opponent_unresolved"] = penalty_opponent_unresolved

    # §25.4: route DFL BallDeflection annotations (woodwork qualifier on the paired shot; referee /
    # unpaired counted). Runs on source-orientation coords, before the LTR flip.
    claim_counts["deflection_counts"] = _route_ball_deflections(events, actions)

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
    # Interleave synth (GK distribution) and F4 derived rows onto the shared *10 _row_order grid, then sort
    # once. A stable mergesort keeps the grid order; the later chronological sort (convert_to_actions) is
    # also stable, so same-timestamp derived rows keep their just-before-partner placement.
    extra_frames = [f for f in (synth, claim_derived) if len(f) > 0]
    if extra_frames:
        actions["_row_order"] = np.arange(len(actions), dtype=np.int64) * 10
        combined = pd.concat([actions, *extra_frames], ignore_index=True, sort=False)
        actions = (
            combined.sort_values("_row_order", kind="mergesort").drop(columns=["_row_order"]).reset_index(drop=True)
        )

    # Drop non_action rows.
    actions = actions[actions["type_id"] != spadlconfig.actiontype_id["non_action"]].reset_index(drop=True)

    if len(actions) == 0:
        return _empty_raw_actions(preserve_native, events), claim_counts

    return actions, claim_counts


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

    # The synthesized distribution (throwOut->pass / punt->goalkick) inherits the parent Play's
    # native completion via the SAME success-allowlist as the main path (single-sourced helpers);
    # empty/absent -> success. Mirrors the open-play / set-piece rule above.
    synth_eval = _extract_play_eval(src)
    result_ids_synth = np.where(
        _play_evaluation_is_fail(synth_eval),
        spadlconfig.result_id["fail"],
        spadlconfig.result_id["success"],
    ).astype(np.int64)
    _warn_unexpected_play_eval(synth_eval)

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
            "result_id": result_ids_synth,
            "bodypart_id": bodypart_ids_synth,
            # Synthetic GK-distribution rows are never shots/crosses -> pd.NA (TF-51 prereq).
            "shot_blocked": _blocked_flag(n_synth),
            "cross_blocked": _blocked_flag(n_synth),
            # Converter-synthesized second half of a GK distribution -> derived (F4 / decision B2).
            "action_provenance": np.full(n_synth, "derived", dtype=object),
        }
    )

    # _row_order: interleave key on a *10 grid shared with the F4 BallClaiming synthesis. Source rows get
    # index*10; a distribution second half gets source_index*10 + 5 so it sorts immediately after its
    # keeper_pick_up; an F4 derived row gets partner_index*10 - 1 (just before its partner).
    synth["_row_order"] = src_indices.astype(np.int64) * 10 + 5

    if preserve_native:
        for col in preserve_native:
            if col in src.columns:
                synth[col] = src[col].to_numpy()
            else:
                synth[col] = np.nan

    return synth


def _ball_claim_report_template() -> dict:
    """Empty F4 ``BallClaiming`` audit counts (returned when no BallClaiming events are present)."""
    return {
        "ball_claim_counts": {},
        "interceptions_added": 0,
        "keeper_pick_ups_added": 0,
        "ball_claim_unattached": 0,
        "penalty_result_method_counts": {},
        "penalty_retakes": 0,
        "deflection_counts": {},
        "own_goal_attribution_conflicts": 0,
        "penalty_opponent_unresolved": 0,
    }


def _synthesize_ball_claiming_actions(
    events: pd.DataFrame,
    actions: pd.DataFrame,
    preserve_native: list[str] | None,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """F4: fold DFL ``BallClaiming`` annotations into the SPADL stream (spec section 7.1, option B).

    ``BallClaiming`` is an annotation, not a touch (it produces no SPADL row by itself). Each one is
    resolved to a PARTNER action -- the same player's nearest SURVIVING SPADL action within +-0.1 s in the
    same period -- and handled per ``claim_type``:

    - ``ball_claim_type`` is recorded as a qualifier on the partner action for ``InterceptedBall`` /
      ``BallHeld`` / ``Block`` (ADR-001 dedicated column); ``BallClaimed`` carries it on its own derived row.
    - ``InterceptedBall`` emits a derived ``interception`` row unless the partner is already a ``tackle`` /
      ``keeper_*`` action (no double count). The row is placed just before the partner (partner timestamp,
      same player, partner start coordinates); with no surviving partner it is a standalone row at the
      claim's own time and coordinates.
    - ``BallClaimed`` is an outfield ball win; spec section 7.1 (amended 2026-10-20) maps it to a derived
      ``interception`` exactly like ``InterceptedBall`` (house precedent: Metrica RECOVERY, SkillCorner
      ``start_type=="recovery"``). Its derived row carries ``ball_claim_type="BallClaimed"`` so ball wins
      stay separable from pass interceptions. Same skip guard and standalone-row rule as ``InterceptedBall``.
    - ``BallHeld`` emits a derived ``keeper_pick_up`` unless a ``keeper_*`` action already exists for that
      player within +-0.1 s (the duplicate-keeper guard). All ``BallHeld`` are by a goalkeeper.
    - ``Block`` sets the qualifier only; never a derived row, never ``keeper_claim``.

    ``actions`` is the main 1:1 frame BEFORE the ``non_action`` drop; partner matching excludes
    ``non_action`` rows explicitly so a dropped partner never attaches. Returns
    ``(ball_claim_type_array, derived_rows, counts)``: the per-main-row ``ball_claim_type`` to assign, a
    derived-rows frame carrying ``_row_order`` (``partner_index*10 - 1``), and the audit counts.
    """
    n = len(actions)
    claim_arr = np.full(n, None, dtype=object)
    counts = _ball_claim_report_template()
    if "event_type" not in events.columns:
        return claim_arr, pd.DataFrame(), counts
    bc = events[events["event_type"] == "BallClaiming"]
    if len(bc) == 0 or n == 0:
        return claim_arr, pd.DataFrame(), counts

    non_action_id = spadlconfig.actiontype_id["non_action"]
    tackle_id = spadlconfig.actiontype_id["tackle"]
    keeper_ids = {
        spadlconfig.actiontype_id["keeper_save"],
        spadlconfig.actiontype_id["keeper_claim"],
        spadlconfig.actiontype_id["keeper_punch"],
        spadlconfig.actiontype_id["keeper_pick_up"],
    }
    skip_ids = keeper_ids | {tackle_id}

    survive = actions["type_id"].to_numpy() != non_action_id
    # ADR-019: canonical id as the dict-key token (never astype(str), which yields '366.0' for a float id).
    a_player = canonical_id_series(actions["player_id"]).to_numpy()
    a_period = actions["period_id"].to_numpy()
    a_time = actions["time_seconds"].to_numpy(dtype=np.float64)
    a_type = actions["type_id"].to_numpy()

    # Index surviving actions by (player, period) once so each claim does a small lookup, not a full scan
    # (ADR-068 no-rescan-in-loop; BallClaiming count is tiny but the index keeps the cost linear).
    from collections import defaultdict

    by_key: dict[tuple[object, int], list[int]] = defaultdict(list)  # key[0] is a canonical id (str | NA)
    for j in range(n):
        if survive[j]:
            by_key[(a_player[j], int(a_period[j]))].append(j)

    derived: list[dict] = []
    for _, r in bc.iterrows():
        ct = r.get("claim_type")
        if ct is None or (isinstance(ct, float) and np.isnan(ct)):
            continue
        ct = str(ct)
        counts["ball_claim_counts"][ct] = counts["ball_claim_counts"].get(ct, 0) + 1
        pid = canonical_id(r["player_id"])  # ADR-019 dict-key token
        per = int(r["period"])
        t = float(r["timestamp_seconds"])
        cands = [j for j in by_key.get((pid, per), ()) if abs(a_time[j] - t) <= 0.1]
        if cands:
            j = min(cands, key=lambda k: abs(a_time[k] - t))
            partner_type = int(a_type[j])
            near_keeper = any(int(a_type[k]) in keeper_ids for k in cands)
            # Record the qualifier on the nearest surviving partner for every claim type EXCEPT BallClaimed.
            # A BallClaimed is itself the ball win; it carries its claim_type on its own derived interception
            # row (below), not on the partner (the claimer's own next touch). See spec section 7.1 (amended).
            if ct != "BallClaimed":
                claim_arr[j] = ct
        else:
            j = None
            partner_type = None
            near_keeper = False

        if ct in ("InterceptedBall", "BallClaimed"):
            # F4 (spec section 7.1, amended 2026-10-20): a BallClaimed is an outfield ball win, mapped to a
            # derived interception exactly like InterceptedBall (house precedent: Metrica RECOVERY ->
            # interception, SkillCorner start_type=="recovery"). The InterceptedBall row carries no
            # ball_claim_type -- its qualifier sits on the intercepted partner -- while the BallClaimed row
            # carries ball_claim_type="BallClaimed", so a consumer keeps ball wins separable from pass
            # interceptions. Both skip when the partner is already a tackle/keeper touch (0 BallClaimed cases
            # on IDSSE) and emit a standalone row at the claim's own time/coords when no partner survives.
            if j is not None and partner_type in skip_ids:
                # Partner already carries the defensive touch -> qualifier only, no derived row. BallClaimed's
                # qualifier was not set in the cands block above (it normally rides its derived row), so set it
                # here -- matching InterceptedBall-skip -- so the claim still lands (closes the "206 land"
                # invariant; 0 such BallClaimed cases on IDSSE). TF46C-IMPL-19 / A1-IMPL-12.
                if ct == "BallClaimed":
                    claim_arr[j] = ct
                continue
            token = "BallClaimed" if ct == "BallClaimed" else None
            derived.append(_build_claim_row(r, actions, j, "interception", preserve_native, ball_claim_type=token))
            counts["interceptions_added"] += 1
        elif ct == "BallHeld":
            if near_keeper:
                continue  # a keeper action already exists -- the duplicate-keeper guard
            derived.append(_build_claim_row(r, actions, j, "keeper_pick_up", preserve_native))
            counts["keeper_pick_ups_added"] += 1
        elif j is None:  # Block with no surviving partner: the qualifier has nowhere to land
            counts["ball_claim_unattached"] += 1

    derived_df = pd.DataFrame(derived) if derived else pd.DataFrame()
    return claim_arr, derived_df, counts


def _build_claim_row(
    claim: pd.Series,
    actions: pd.DataFrame,
    partner_idx: int | None,
    spadl_type: str,
    preserve_native: list[str] | None,
    ball_claim_type: str | None = None,
) -> dict:
    """Build one derived F4 row (``interception`` / ``keeper_pick_up``).

    With a partner it inherits the partner action's (game, period, time, team, player, start coordinates)
    and sorts just before it (``_row_order = partner_index*10 - 1``); without one it uses the claim event's
    own fields and sorts after the main grid (its own timestamp drives its final chronological position).

    ``ball_claim_type`` is set on the row itself only for a BallClaimed-origin interception (so ball wins
    stay separable from pass interceptions); it is None (omitted) for every other derived row.
    """
    bodypart = spadlconfig.bodypart_id["other" if spadl_type == "keeper_pick_up" else "foot"]
    # Only non-NA fields are set here. The remaining all-NA columns (shot_blocked / cross_blocked / tackle_* /
    # type_source, and ball_claim_type when None below) are omitted so the downstream pd.concat fills them
    # from the main frame without the "concatenation with empty or all-NA entries" FutureWarning -- matching
    # the GK-distribution synth frame, which likewise omits columns it never populates.
    base: dict = {
        "type_id": spadlconfig.actiontype_id[spadl_type],
        "result_id": spadlconfig.result_id["success"],
        "bodypart_id": bodypart,
        "action_provenance": "derived",
    }
    if ball_claim_type is not None:
        base["ball_claim_type"] = ball_claim_type
    suffix = "_f4_interception" if spadl_type == "interception" else "_f4_keeper_pick_up"
    base["original_event_id"] = str(claim["event_id"]) + suffix
    if partner_idx is not None:
        p = actions.iloc[partner_idx]
        base["game_id"] = p["game_id"]
        base["period_id"] = int(p["period_id"])
        base["time_seconds"] = float(p["time_seconds"])
        base["team_id"] = p["team_id"]
        base["player_id"] = p["player_id"]
        base["start_x"] = float(p["start_x"])
        base["start_y"] = float(p["start_y"])
        base["_row_order"] = int(partner_idx) * 10 - 1
    else:
        base["game_id"] = claim["match_id"]
        base["period_id"] = int(claim["period"])
        base["time_seconds"] = float(claim["timestamp_seconds"])
        base["team_id"] = claim["team"]
        base["player_id"] = claim["player_id"]
        base["start_x"] = float(claim["x"])
        base["start_y"] = float(claim["y"])
        base["_row_order"] = len(actions) * 10 + 1_000_000  # after the grid; chronological sort re-places it
    base["end_x"] = base["start_x"]
    base["end_y"] = base["start_y"]
    if preserve_native:
        for col in preserve_native:
            base[col] = np.nan
    return base


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
