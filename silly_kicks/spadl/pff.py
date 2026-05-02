"""PFF FC / Gradient Sports DataFrame SPADL converter.

Converts already-flattened PFF events DataFrames (e.g., produced from the
public WC 2022 release JSON via ``pd.json_normalize`` + a roster join) to
SPADL actions.

PFF event vocabulary (recognized as input)
------------------------------------------

PFF events have a hierarchical shape: ``gameEvents`` envelope (high-level
game-event class: OTB / OUT / SUB / FIRSTKICKOFF / SECONDKICKOFF / THIRDKICKOFF /
FOURTHKICKOFF / END / FOUL / OFF / ON / G) + ``possessionEvents`` payload
(detailed possession-event class: PA / SH / CR / CL / BC / CH / RE / TC / IT
/ FO) + per-event ``fouls`` dict. Each top-level event JSON list element
flattens to one row in the events DataFrame consumed here. (Note: ``fouls``
is a single dict per event in real PFF data, not a JSON array.)

The converter dispatches on the tuple ``(game_event_type,
possession_event_type, set_piece_type)``. See the spec at
``docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md`` § 4.4
for the full mapping table.

Coordinate system
-----------------

PFF source coordinates are pitch-centered meters (origin at center spot,
x ∈ ~[-52.5, 52.5], y ∈ ~[-34, 34]). The converter translates to SPADL's
bottom-left-origin meters (x ∈ [0, 105], y ∈ [0, 68]) and applies per-period
direction-of-play normalization so all teams attack left-to-right (the
standard SPADL invariant).

PFF coordinates reflect actual on-field direction (which switches between
periods); the converter therefore requires explicit direction parameters
(``home_team_start_left`` and, when ET is present, ``home_team_start_left_extratime``).
Both come from PFF metadata JSON (``homeTeamStartLeft``, ``homeTeamStartLeftExtraTime``).

ADR-001: identifier conventions are sacred (silly-kicks 2.0.0)
---------------------------------------------------------------

The converter never overrides ``team_id`` / ``player_id`` from the
on-the-ball actor (PFF ``gameEvents.playerId``). Tackle winner/loser
qualifier values (PFF ``challenge_winner_player_id`` / ``challenger_player_id``)
surface via dedicated output columns:

==========================  ============================================
Output column               PFF qualifier source
==========================  ============================================
``tackle_winner_player_id`` ``challenge_winner_player_id``
``tackle_winner_team_id``   ``challenge_winner_team_id``  (caller-supplied via roster join)
``tackle_loser_player_id``  derived: challenger_player_id if winner != challenger
                              else event row's player_id
``tackle_loser_team_id``    derived: same logic on team_id
==========================  ============================================

The output schema is :data:`silly_kicks.spadl.PFF_SPADL_COLUMNS`
(extends :data:`silly_kicks.spadl.SPADL_COLUMNS` with the 4 tackle columns).
"""

from collections import Counter

import numpy as np
import pandas as pd

from silly_kicks.tracking import _direction

from . import config as spadlconfig
from .orientation import PER_PERIOD_ABSOLUTE, to_spadl_ltr, validate_input_convention
from .schema import PFF_SPADL_COLUMNS, ConversionReport
from .utils import _finalize_output, _validate_input_columns, _validate_preserve_native

# ---------------------------------------------------------------------------
# Required input columns (raise ValueError if any are missing)
# ---------------------------------------------------------------------------
EXPECTED_INPUT_COLUMNS: frozenset[str] = frozenset(
    {
        # Identification & timing
        "game_id",
        "event_id",
        "possession_event_id",
        "period_id",
        "time_seconds",
        "team_id",
        "player_id",
        # Event-class dispatch keys
        "game_event_type",
        "possession_event_type",
        "set_piece_type",
        # Ball position (PFF centered meters)
        "ball_x",
        "ball_y",
        # Body part / pass / cross qualifiers
        "body_type",
        "ball_height_type",
        "pass_outcome_type",
        "pass_type",
        "incompletion_reason_type",
        "cross_outcome_type",
        "cross_type",
        "cross_zone_type",
        # Shot qualifiers
        "shot_outcome_type",
        "shot_type",
        "shot_nature_type",
        "shot_initial_height_type",
        "save_height_type",
        "save_rebound_type",
        # Carry / dribble qualifiers
        "carry_type",
        "ball_carry_outcome",
        "carry_intent",
        "carry_defender_player_id",
        # Challenge / tackle qualifiers (PFF carries actor IDs only as players;
        # caller supplies team affiliation via roster join — see § 4.5 of spec)
        "challenge_type",
        "challenge_outcome_type",
        "challenger_player_id",
        "challenger_team_id",
        "challenge_winner_player_id",
        "challenge_winner_team_id",
        "tackle_attempt_type",
        # Clearance / rebound / GK / touch qualifiers
        "clearance_outcome_type",
        "rebound_outcome_type",
        "keeper_touch_type",
        "touch_outcome_type",
        "touch_type",
        # Foul (one PFF event row has at most one fouls[0] entry; flatten)
        "foul_type",
        "on_field_offense_type",
        "final_offense_type",
        "on_field_foul_outcome_type",
        "final_foul_outcome_type",
    }
)


# ---------------------------------------------------------------------------
# Vectorized dispatch helpers
# ---------------------------------------------------------------------------
def _dispatch_bodypart(body_type: pd.Series) -> np.ndarray:
    """Map PFF body_type codes to SPADL bodypart_id (vectorized).

    Mapping: L → foot_left, R → foot_right, H → head, O → other, null → foot.
    """
    mapping: dict[object, str] = {
        "L": "foot_left",
        "R": "foot_right",
        "H": "head",
        "O": "other",
    }
    name_series = body_type.map(mapping).fillna("foot")
    return name_series.map(spadlconfig.bodypart_id).astype("int64").to_numpy()


def _dispatch_actiontype_resultid(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized dispatch from
    ``(game_event_type, possession_event_type, set_piece_type)`` to
    ``(type_id, result_id)`` per row.

    type_id defaults to ``non_action``; result_id defaults to ``fail``.
    Refinement passes (RE → keeper_pick_up) follow.
    """
    ge = events["game_event_type"].fillna("").to_numpy()
    pe = events["possession_event_type"].fillna("").to_numpy()
    sp = events["set_piece_type"].fillna("").to_numpy()
    pass_outcome = events["pass_outcome_type"].fillna("").to_numpy()
    cross_outcome = events["cross_outcome_type"].fillna("").to_numpy()
    shot_outcome = events["shot_outcome_type"].fillna("").to_numpy()
    foul_outcome = events["final_foul_outcome_type"].fillna("").to_numpy()

    at_ids = spadlconfig.actiontype_id
    rs_ids = spadlconfig.result_id

    # type_id dispatch — top-down priority (np.select picks first match).
    type_conds = [
        (ge == "OTB") & (pe == "PA") & (sp == "F"),
        (ge == "OTB") & (pe == "PA") & (sp == "C"),
        (ge == "OTB") & (pe == "PA") & (sp == "T"),
        (ge == "OTB") & (pe == "PA") & (sp == "G"),
        (ge == "OTB") & (pe == "PA"),  # PA + O / K / unknown → pass
        (ge == "OTB") & (pe == "CR") & (sp == "F"),
        (ge == "OTB") & (pe == "CR") & (sp == "C"),
        (ge == "OTB") & (pe == "CR"),  # CR + O / others → cross
        (ge == "OTB") & (pe == "SH") & (sp == "F"),
        (ge == "OTB") & (pe == "SH") & (sp == "P"),
        (ge == "OTB") & (pe == "SH"),  # SH + O / K / others → shot
        (ge == "OTB") & (pe == "CL"),
        (ge == "OTB") & (pe == "BC"),
        (ge == "OTB") & (pe == "CH"),
        (ge == "OTB") & (pe == "RE"),
        (ge == "OTB") & (pe == "TC"),
    ]
    type_choices = [
        at_ids["freekick_short"],
        at_ids["corner_short"],
        at_ids["throw_in"],
        at_ids["goalkick"],
        at_ids["pass"],
        at_ids["freekick_crossed"],
        at_ids["corner_crossed"],
        at_ids["cross"],
        at_ids["shot_freekick"],
        at_ids["shot_penalty"],
        at_ids["shot"],
        at_ids["clearance"],
        at_ids["dribble"],
        at_ids["tackle"],
        at_ids["keeper_save"],  # default for RE; refined for keeper_pick_up below
        at_ids["bad_touch"],
    ]
    type_id_arr = np.select(type_conds, type_choices, default=at_ids["non_action"]).astype("int64")

    # Refinement: RE rows with catch-class keeper_touch_type → keeper_pick_up.
    keeper_touch = events["keeper_touch_type"].fillna("").to_numpy()
    catch_class: set[str] = {"C"}
    is_catch = (pe == "RE") & np.isin(keeper_touch, list(catch_class))
    type_id_arr = np.where(is_catch, at_ids["keeper_pick_up"], type_id_arr).astype("int64")

    # result_id dispatch.
    is_pass_class = (pe == "PA") | (pe == "CR")
    pass_success = is_pass_class & ((pass_outcome == "C") | (cross_outcome == "C"))  # noqa: S105

    is_shot = pe == "SH"
    shot_goal = is_shot & (shot_outcome == "G")
    shot_owngoal = is_shot & (shot_outcome == "O")

    is_yellow = pd.Series(foul_outcome).str.startswith(("Y", "2Y")).fillna(False).to_numpy()
    is_red = pd.Series(foul_outcome).str.startswith(("R", "SR")).fillna(False).to_numpy()

    result_conds = [pass_success, shot_goal, shot_owngoal, is_yellow, is_red]
    result_choices = [
        rs_ids["success"],
        rs_ids["success"],
        rs_ids["owngoal"],
        rs_ids["yellow_card"],
        rs_ids["red_card"],
    ]
    result_id_arr = np.select(result_conds, result_choices, default=rs_ids["fail"]).astype("int64")

    return type_id_arr, result_id_arr


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert a flattened PFF events DataFrame to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        PFF-shaped events DataFrame. Required columns:
        :data:`EXPECTED_INPUT_COLUMNS`.
    home_team_id : int
        PFF home-team identifier (``homeTeam.id`` from PFF metadata JSON).
    home_team_start_left : bool
        Whether the home team attacks toward the left goal in period 1
        (``homeTeamStartLeft`` from PFF metadata JSON). Drives per-period
        direction-of-play normalization.
    home_team_start_left_extratime : bool or None, default None
        Same flag for ET periods 3/4 (``homeTeamStartLeftExtraTime`` from
        PFF metadata JSON). Required only if the events span ET; raises
        ``ValueError`` if events have ``period_id`` ∈ {3, 4} but this is
        ``None``.
    preserve_native : list[str] or None, default None
        Optional input columns to passthrough into the output unchanged.

    Returns
    -------
    tuple[pd.DataFrame, ConversionReport]
        SPADL actions matching :data:`silly_kicks.spadl.PFF_SPADL_COLUMNS`
        and a ConversionReport audit trail.

    Raises
    ------
    ValueError
        If any column in :data:`EXPECTED_INPUT_COLUMNS` is missing from
        ``events``, or if ``period_id`` 3/4 rows exist but
        ``home_team_start_left_extratime`` is ``None``.

    Examples
    --------
    Convert a single match's events to SPADL::

        from silly_kicks.spadl import pff
        actions, report = pff.convert_to_actions(
            events,
            home_team_id=366,             # Netherlands in WC 2022 NED-USA
            home_team_start_left=True,    # from match metadata
        )
        assert not report.has_unrecognized
    """
    _validate_input_columns(events, set(EXPECTED_INPUT_COLUMNS), provider="PFF")
    _validate_preserve_native(events, preserve_native, provider="PFF", schema=PFF_SPADL_COLUMNS)

    total_events_input = len(events)

    # PR-S23 / silly-kicks 3.0.1: validator re-enabled after TF-22 detector
    # hardening. PFF events ship PER_PERIOD_ABSOLUTE; the detector now
    # correctly defers (convention=None) on sparse-shot matches rather than
    # false-positiving ABSOLUTE_FRAME_HOME_RIGHT. ball_x is centered (-52.5
    # to +52.5); shift to 0-105 frame for the detector's high-x/low-x logic.
    if "ball_x" in events.columns and "team_id" in events.columns and "period_id" in events.columns:
        _detector_input = events.assign(
            _sk_ball_x_shifted=events["ball_x"].astype("float64") + 52.5,
            _sk_is_shot=(events["possession_event_type"].fillna("") == "SH"),
        )
        validate_input_convention(
            _detector_input,
            declared=PER_PERIOD_ABSOLUTE,
            match_col="game_id",
            x_col="_sk_ball_x_shifted",
            team_col="team_id",
            period_col="period_id",
            is_shot_col="_sk_is_shot",
            x_max=spadlconfig.field_length,
        )

    # ------------------------------------------------------------------
    # Per-period direction lookup (ET fallback validation)
    # ------------------------------------------------------------------
    if events["period_id"].isin([3, 4]).any() and home_team_start_left_extratime is None:
        raise ValueError(
            "PFF convert_to_actions: events contain ET periods (period_id ∈ {3, 4}) "
            "but home_team_start_left_extratime was not provided. Set it explicitly to "
            "match metadata.homeTeamStartLeftExtraTime, or filter ET events out before calling."
        )
    home_attacks_right_per_period = _direction.home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )

    # ------------------------------------------------------------------
    # Exclusion filtering — drop rows whose
    # (game_event_type, possession_event_type) pair is in the documented
    # excluded set; tally counts for the ConversionReport.
    # ------------------------------------------------------------------
    ge_arr_full = events["game_event_type"].fillna("").to_numpy()
    pe_arr_full = events["possession_event_type"].fillna("").to_numpy()

    # Excluded game_event_types: structural / metadata events with no SPADL
    # counterpart. Includes OFF (player-off-field), ON (player-on-field), G
    # (game-marker — null actor), and the four kickoff variants (each period's
    # restart). Empirically validated against the full WC 2022 dataset.
    excluded_ge_types = {
        "OUT",
        "SUB",
        "FIRSTKICKOFF",
        "SECONDKICKOFF",
        "THIRDKICKOFF",
        "FOURTHKICKOFF",
        "END",
        "OFF",
        "ON",
        "G",
    }
    # Excluded (game_event_type, possession_event_type) pairs:
    # - ("OTB", "IT"): ball-receipt (analog of StatsBomb "Ball Receipt*").
    # - ("OTB", ""):   initialNonEvent markers — OTB rows with empty PE
    #                  carry initialNonEvent=true and have no SPADL semantics.
    excluded_pair_keys = {("OTB", "IT"), ("OTB", "")}

    is_excluded_ge = np.isin(ge_arr_full, list(excluded_ge_types))
    is_excluded_pair = np.zeros(len(events), dtype=bool)
    for ge_, pe_ in excluded_pair_keys:
        is_excluded_pair |= (ge_arr_full == ge_) & (pe_arr_full == pe_)
    is_excluded = is_excluded_ge | is_excluded_pair

    excluded_counts: Counter = Counter()
    for ge_t in ge_arr_full[is_excluded_ge]:
        excluded_counts[str(ge_t)] += 1
    for ge_, pe_ in excluded_pair_keys:
        n = int(((ge_arr_full == ge_) & (pe_arr_full == pe_)).sum())
        if n > 0:
            excluded_counts[f"{ge_}+{pe_}"] = n

    events = events.loc[~is_excluded].reset_index(drop=True)

    # Empty-input fast path (after exclusion): empty schema-compliant output.
    if len(events) == 0:
        actions = pd.DataFrame({col: [] for col in PFF_SPADL_COLUMNS.keys()})
        for col in (
            "tackle_winner_player_id",
            "tackle_winner_team_id",
            "tackle_loser_player_id",
            "tackle_loser_team_id",
        ):
            actions[col] = pd.array([], dtype="Int64")
        actions = _finalize_output(actions, schema=PFF_SPADL_COLUMNS)
        report = ConversionReport(
            provider="PFF",
            total_events=total_events_input,
            total_actions=0,
            mapped_counts={},
            excluded_counts=dict(excluded_counts),
            unrecognized_counts={},
        )
        return actions, report

    # ------------------------------------------------------------------
    # Dispatch (type_id, result_id, bodypart_id)
    # ------------------------------------------------------------------
    type_id_arr, result_id_arr = _dispatch_actiontype_resultid(events)
    bodypart_id_arr = _dispatch_bodypart(events["body_type"])

    # ------------------------------------------------------------------
    # Coordinate translation (PFF centered → SPADL bottom-left meters)
    # ------------------------------------------------------------------
    actions = pd.DataFrame(
        {
            "game_id": events["game_id"].astype("int64").values,
            "original_event_id": events["event_id"].astype("object").values,
            "action_id": np.arange(len(events), dtype="int64"),
            "period_id": events["period_id"].astype("int64").values,
            "time_seconds": events["time_seconds"].astype("float64").values,
            "team_id": events["team_id"].astype("int64").values,
            "player_id": (events["player_id"].astype("Int64").fillna(0).astype("int64").values),
            "start_x": (events["ball_x"].astype("float64") + 52.5).values,
            "start_y": (events["ball_y"].astype("float64") + 34.0).values,
            "end_x": (events["ball_x"].astype("float64") + 52.5).values,
            "end_y": (events["ball_y"].astype("float64") + 34.0).values,
            "type_id": type_id_arr,
            "result_id": result_id_arr,
            "bodypart_id": bodypart_id_arr,
            "tackle_winner_player_id": pd.array([pd.NA] * len(events), dtype="Int64"),
            "tackle_winner_team_id": pd.array([pd.NA] * len(events), dtype="Int64"),
            "tackle_loser_player_id": pd.array([pd.NA] * len(events), dtype="Int64"),
            "tackle_loser_team_id": pd.array([pd.NA] * len(events), dtype="Int64"),
        }
    )

    # ------------------------------------------------------------------
    # Tackle winner/loser passthrough (ADR-001)
    # ------------------------------------------------------------------
    is_tackle = np.asarray(
        events["possession_event_type"].fillna("") == "CH",
        dtype=bool,
    )
    if is_tackle.any():
        # Restrict every right-hand-side array to the tackle rows so the
        # length matches the .loc[mask, ...] left-hand side exactly.
        tackle_events = events.loc[is_tackle]
        winner_pid = tackle_events["challenge_winner_player_id"].astype("Int64")
        winner_tid = tackle_events["challenge_winner_team_id"].astype("Int64")
        challenger_pid = tackle_events["challenger_player_id"].astype("Int64")
        challenger_tid = tackle_events["challenger_team_id"].astype("Int64")
        event_pid = tackle_events["player_id"].astype("Int64")
        event_tid = tackle_events["team_id"].astype("Int64")

        challenger_won = np.asarray((winner_pid == challenger_pid).fillna(False), dtype=bool)
        loser_pid = pd.array(  # type: ignore[reportCallIssue]
            np.where(challenger_won, event_pid.to_numpy(), challenger_pid.to_numpy()),
            dtype="Int64",
        )
        loser_tid = pd.array(  # type: ignore[reportCallIssue]
            np.where(challenger_won, event_tid.to_numpy(), challenger_tid.to_numpy()),
            dtype="Int64",
        )

        tackle_mask = pd.Series(is_tackle, index=actions.index)
        actions.loc[tackle_mask, "tackle_winner_player_id"] = winner_pid.to_numpy()
        actions.loc[tackle_mask, "tackle_winner_team_id"] = winner_tid.to_numpy()
        actions.loc[tackle_mask, "tackle_loser_player_id"] = loser_pid
        actions.loc[tackle_mask, "tackle_loser_team_id"] = loser_tid

    # ------------------------------------------------------------------
    # Foul row handling (two paths, depending on parent dispatch result)
    #
    # PFF places foul info in a per-event ``fouls`` dict that may co-occur with:
    #   - A real possession event (e.g. PA / CR / SH with a foul committed
    #     during it) -- we synthesize an ADDITIONAL foul row alongside the
    #     parent action.
    #   - A dedicated FOUL gameEventType (possessionEventType="FO") -- the
    #     parent dispatched to non_action because no PA/CR/SH/etc. row matches.
    #     We convert IN-PLACE so the foul is the canonical action row (avoids
    #     phantom non_action rows in the output).
    #
    # The dispatch table doesn't know about the inline foul info; this block
    # is the single source of truth for foul-row creation.
    # ------------------------------------------------------------------
    foul_mask = np.asarray(events["foul_type"].notna(), dtype=bool)
    non_action_id = spadlconfig.actiontype_id["non_action"]
    foul_id = spadlconfig.actiontype_id["foul"]

    # Pre-compute card-result vector once, indexed per event row.
    foul_outcome_full = events["final_foul_outcome_type"].fillna("").to_numpy()
    is_yellow_full = np.asarray(
        pd.Series(foul_outcome_full).str.startswith(("Y", "2Y")).fillna(False),
        dtype=bool,
    )
    is_red_full = np.asarray(
        pd.Series(foul_outcome_full).str.startswith(("R", "SR")).fillna(False),
        dtype=bool,
    )
    foul_result_full = np.where(
        is_yellow_full,
        spadlconfig.result_id["yellow_card"],
        np.where(is_red_full, spadlconfig.result_id["red_card"], spadlconfig.result_id["success"]),
    ).astype("int64")

    in_place_mask = foul_mask & (actions["type_id"].to_numpy() == non_action_id)
    synth_mask = foul_mask & ~in_place_mask

    # Convert in-place: dedicated FOUL events become the canonical foul row.
    if in_place_mask.any():
        actions.loc[in_place_mask, "type_id"] = foul_id
        actions.loc[in_place_mask, "result_id"] = foul_result_full[in_place_mask]
        actions.loc[in_place_mask, "bodypart_id"] = spadlconfig.bodypart_id["foot"]

    # Synthesize additional row: parent already dispatched to a real action.
    if synth_mask.any():
        synth_rows = actions.loc[synth_mask].copy()
        synth_rows["type_id"] = foul_id
        synth_rows["result_id"] = foul_result_full[synth_mask]
        synth_rows["bodypart_id"] = spadlconfig.bodypart_id["foot"]
        # Insert synthesized rows immediately AFTER their parents via .5-offset
        # sort key, then renumber action_id dense.
        actions["__order__"] = np.arange(len(actions), dtype="float64")
        synth_rows["__order__"] = np.arange(len(actions))[synth_mask] + 0.5
        actions = pd.concat([actions, synth_rows], ignore_index=True)
        actions = actions.sort_values("__order__").reset_index(drop=True)
        actions = actions.drop(columns="__order__")
        actions["action_id"] = np.arange(len(actions), dtype="int64")

    # ------------------------------------------------------------------
    # Per-period direction-of-play normalisation. Routed through the canonical
    # to_spadl_ltr dispatcher per ADR-006 (silly-kicks 3.0.0); behaviour
    # preserved exactly because the dispatcher's PER_PERIOD_ABSOLUTE branch
    # uses the same home_attacks_right_per_period mapping that PFF computed
    # from metadata flags above.
    # ------------------------------------------------------------------
    actions = to_spadl_ltr(
        actions,
        input_convention=PER_PERIOD_ABSOLUTE,
        home_team_id=home_team_id,
        home_attacks_right_per_period=home_attacks_right_per_period,
    )

    # ------------------------------------------------------------------
    # end_x / end_y from next-action start_x/y within same period.
    # Last row of each period falls back to its own start_x/y.
    # ------------------------------------------------------------------
    if len(actions) > 0:
        next_start_x = actions["start_x"].shift(-1)
        next_start_y = actions["start_y"].shift(-1)
        same_period = actions["period_id"].eq(actions["period_id"].shift(-1))
        actions["end_x"] = np.where(
            same_period & next_start_x.notna(),
            next_start_x,
            actions["start_x"],
        )
        actions["end_y"] = np.where(
            same_period & next_start_y.notna(),
            next_start_y,
            actions["start_y"],
        )

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    actions = _finalize_output(
        actions,
        schema=PFF_SPADL_COLUMNS,
        extra_columns=preserve_native,
    )

    # ------------------------------------------------------------------
    # ConversionReport: mapped_counts uses SPADL action-type names;
    # unrecognized_counts surfaces (ge, pe) pairs that mapped to non_action.
    # ------------------------------------------------------------------
    id_to_name = {i: name for i, name in enumerate(spadlconfig.actiontypes)}
    mapped_counts: Counter = Counter()
    for tid in actions["type_id"].to_numpy():
        name = id_to_name.get(int(tid), "non_action")
        if name == "non_action":
            continue
        mapped_counts[name] += 1

    # Unrecognized-counts: any (ge, pe) pair that landed in non_action AND
    # was not already absorbed by foul-row handling (in-place conversion or
    # synthesis). Computed from the dispatch result + foul_mask, not from a
    # post-synthesis lookup -- PFF gameEventId is not row-unique (multiple
    # rows can share a gameEventId when a high-level event has nested
    # possession events), so a lookup-based approach yields cross-talk.
    unrecognized_counts: Counter = Counter()
    unrecognized_mask = (type_id_arr == non_action_id) & ~foul_mask
    if unrecognized_mask.any():
        ge_filtered = events["game_event_type"].fillna("").to_numpy()
        pe_filtered = events["possession_event_type"].fillna("").to_numpy()
        for ge_v, pe_v in zip(
            ge_filtered[unrecognized_mask],
            pe_filtered[unrecognized_mask],
            strict=True,
        ):
            unrecognized_counts[f"{ge_v}+{pe_v}"] += 1

    report = ConversionReport(
        provider="PFF",
        total_events=total_events_input,
        total_actions=len(actions),
        mapped_counts=dict(mapped_counts),
        excluded_counts=dict(excluded_counts),
        unrecognized_counts=dict(unrecognized_counts),
    )
    return actions, report
