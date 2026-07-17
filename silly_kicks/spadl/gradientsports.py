"""Gradient Sports (formerly PFF FC) DataFrame SPADL converter.

Converts already-flattened Gradient Sports events DataFrames (e.g., produced
from the public WC 2022 release JSON via ``pd.json_normalize`` + a roster
join) to SPADL actions.

Event vocabulary (recognized as input)
--------------------------------------

Gradient Sports events have a hierarchical shape: ``gameEvents`` envelope
(high-level game-event class: OTB / OUT / SUB / FIRSTKICKOFF / SECONDKICKOFF /
THIRDKICKOFF / FOURTHKICKOFF / END / FOUL / OFF / ON / G) +
``possessionEvents`` payload (detailed possession-event class: PA / SH / CR /
CL / BC / CH / RE / TC / IT / FO) + per-event ``fouls`` dict. Each top-level
event JSON list element flattens to one row in the events DataFrame consumed
here. (Note: ``fouls`` is a single dict per event, not a JSON array.)

The converter dispatches on the tuple ``(game_event_type,
possession_event_type, set_piece_type)``. See the spec at
``docs/superpowers/specs/2026-04-30-pff-fc-events-converter-design.md`` § 4.4
for the full mapping table.

Coordinate system
-----------------

Source coordinates are pitch-centered meters (origin at center spot,
x ∈ ~[-52.5, 52.5], y ∈ ~[-34, 34]). The converter translates to SPADL's
bottom-left-origin meters (x ∈ [0, 105], y ∈ [0, 68]) and applies per-period
direction-of-play normalization so all teams attack left-to-right (the
standard SPADL invariant).

Coordinates reflect actual on-field direction (which switches between
periods); the converter therefore requires explicit direction parameters
(``home_team_start_left`` and, when ET is present, ``home_team_start_left_extratime``).
Both come from metadata JSON (``homeTeamStartLeft``, ``homeTeamStartLeftExtraTime``).

ADR-001: identifier conventions are sacred (silly-kicks 2.0.0)
---------------------------------------------------------------

The converter never overrides ``team_id`` / ``player_id`` from the
on-the-ball actor (``gameEvents.playerId``). Tackle winner/loser
qualifier values (``challenge_winner_player_id`` / ``challenger_player_id``)
surface via dedicated output columns.

``team_id`` / ``player_id`` mirror the canonical ``gameEvents`` actor and are
nullable ``Int64``: on the null-actor duel/foul events (``OTB``+``CH``
challenges — a two-sided 50/50 duel with no single owning team — and dedicated
``FOUL``+``FO`` fouls) both ``gameEvents.teamId`` AND ``playerId`` are NULL, so
both output ids are ``NaN`` — NEVER a sentinel ``0`` (a non-NaN sentinel
masquerades as a real id and crashes downstream id-resolution; lakehouse
outage 2026-06-11). Synthesizing ``team_id`` from the duel/foul *qualifiers* is
exactly the ADR-001 violation this contract forbids; the qualifier teams
remain in the dedicated columns below:

==========================  ============================================
Output column               Qualifier source
==========================  ============================================
``tackle_winner_player_id`` ``challenge_winner_player_id``
``tackle_winner_team_id``   ``challenge_winner_team_id``  (caller-supplied via roster join)
``tackle_loser_player_id``  derived: challenger_player_id if winner != challenger
                              else event row's player_id
``tackle_loser_team_id``    derived: same logic on team_id
==========================  ============================================

The output schema is :data:`silly_kicks.spadl.GRADIENTSPORTS_SPADL_COLUMNS`
(extends :data:`silly_kicks.spadl.SPADL_COLUMNS` with the 4 tackle columns).
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd

from silly_kicks.tracking import direction

from . import config as spadlconfig
from .base import _derive_end_coordinates
from .orientation import PER_PERIOD_ABSOLUTE, to_spadl_ltr, validate_input_convention
from .schema import GRADIENTSPORTS_SPADL_COLUMNS, ConversionReport
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
        # Ball position (centered meters)
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
        # Challenge / tackle qualifiers (carries actor IDs only as players;
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
        # Foul (one event row has at most one fouls[0] entry; flatten)
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
    """Map Gradient Sports body_type codes to SPADL bodypart_id (vectorized).

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

    # Component 1 (ADR-0NN): RE + shotOutcome "G" is an OWN GOAL -> bad_touch + owngoal. Provisional
    # here; the post-LTR geometry tripwire in convert_to_actions validates/reverts. Priority over the
    # RE -> keeper_save/keeper_pick_up handling. Scorer (gameEvents.playerId = rebounderPlayerId) and
    # conceding team are kept unchanged (ADR-001); the owngoal RESULT carries credit-the-opponent.
    is_owngoal = (pe == "RE") & (shot_outcome == "G")
    type_id_arr = np.where(is_owngoal, at_ids["bad_touch"], type_id_arr).astype("int64")

    # result_id dispatch.
    is_pass_class = (pe == "PA") | (pe == "CR")
    pass_success = is_pass_class & ((pass_outcome == "C") | (cross_outcome == "C"))  # noqa: S105

    is_shot = pe == "SH"
    shot_goal = is_shot & (shot_outcome == "G")
    # shot_outcome_type "O" is OFF-TARGET (not own-goal). The four main shot
    # outcomes are G=goal / S=saved / O=off-target / B=blocked; only "G" is a
    # success. Verified against the full PFF FC / Gradient Sports WC2022 feed (64
    # matches): "G" counts reproduce every scoreline + shootout tally, while a 0-0
    # match (MAR-ESP) carries O=10 and "O" recurs 4-17x every match — impossible
    # for own goals. Own goals surface under "G"; "shot_outcome_type" alone cannot
    # distinguish them, so the converter maps NO shot outcome to `owngoal` (correct
    # own-goal attribution is an open item pending the PFF FC codebook). All non-"G"
    # shot outcomes fall through to the `fail` default below.

    is_yellow = pd.Series(foul_outcome).str.startswith(("Y", "2Y")).fillna(False).to_numpy()
    is_red = pd.Series(foul_outcome).str.startswith(("R", "SR")).fillna(False).to_numpy()

    result_conds = [pass_success, shot_goal, is_owngoal, is_yellow, is_red]
    result_choices = [
        rs_ids["success"],
        rs_ids["success"],
        rs_ids["owngoal"],
        rs_ids["yellow_card"],
        rs_ids["red_card"],
    ]
    result_id_arr = np.select(result_conds, result_choices, default=rs_ids["fail"]).astype("int64")

    return type_id_arr, result_id_arr


def _resolve_team_ids(events: pd.DataFrame) -> pd.arrays.IntegerArray:
    """Resolve per-row ``team_id`` as nullable ``Int64`` (ADR-001-compliant).

    ``team_id`` mirrors the canonical ``gameEvents`` actor verbatim. On the
    Gradient Sports feed the actor is genuinely absent on the *null-actor* duel
    and foul events (``OTB``+``CH`` challenges — a two-sided 50/50 duel with no
    single owning team — and dedicated ``FOUL``+``FO`` fouls); both
    ``gameEvents.teamId`` AND ``gameEvents.playerId`` are NULL there. Those rows
    MUST carry ``NaN`` ``team_id``, never a sentinel ``0`` (a non-NaN sentinel
    masquerades as a real team id, bypasses every downstream ``pd.isna``
    NaN-route, and crashes the strict opponent-resolution guard).

    Self-heal (ADR-001-legal canonical→canonical): where a row has a real
    ``player_id`` (the canonical actor) but a NULL ``team_id``, derive the team
    from that player's other same-match rows — a player belongs to exactly one
    team per match. This keys ONLY on the canonical ``player_id`` column, NEVER
    on a duel/foul *qualifier* (``challenger`` / ``winner`` / ``culprit``);
    synthesizing ``team_id`` from a qualifier is the ADR-001 violation this
    contract exists to prevent. An ambiguous mapping (a player attributed to >1
    team in the match) raises rather than guesses. Rows with no canonical player
    stay ``NaN``.
    """
    team = events["team_id"].astype("Int64").reset_index(drop=True)
    player = events["player_id"].astype("Int64").reset_index(drop=True)

    needs_fill = (team.isna() & player.notna()).to_numpy()
    if not needs_fill.any():
        return team.array  # type: ignore[return-value]

    needed_players = set(player[needs_fill].dropna().tolist())
    attributed = pd.DataFrame({"player_id": player[team.notna()], "team_id": team[team.notna()]}).dropna()
    attributed = attributed[attributed["player_id"].isin(needed_players)]
    if attributed.empty:
        return team.array  # type: ignore[return-value]  # nothing resolvable; rows stay NaN

    teams_per_player = attributed.groupby("player_id")["team_id"].nunique()
    ambiguous = sorted(teams_per_player[teams_per_player > 1].index.tolist())
    if ambiguous:
        raise ValueError(
            "gradientsports: ambiguous canonical player->team mapping — player_id(s) "
            f"{ambiguous} attributed to more than one team in the same match; refusing "
            "to guess team_id (the ADR-001 self-heal requires a unique team per player)."
        )
    mapping = attributed.drop_duplicates("player_id").set_index("player_id")["team_id"]
    fill_vals = player[needs_fill].map(mapping)  # NaN where player_id not in the map
    team.loc[needs_fill] = fill_vals.to_numpy()
    return team.array  # type: ignore[return-value]


def convert_to_actions(
    events: pd.DataFrame,
    home_team_id: int,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
    preserve_native: list[str] | None = None,
) -> tuple[pd.DataFrame, ConversionReport]:
    """Convert a flattened Gradient Sports events DataFrame to SPADL actions.

    Parameters
    ----------
    events : pd.DataFrame
        Gradient Sports events DataFrame. Required columns:
        :data:`EXPECTED_INPUT_COLUMNS`.
    home_team_id : int
        Home-team identifier (``homeTeam.id`` from metadata JSON).
    home_team_start_left : bool
        Whether the home team attacks toward the left goal in period 1
        (``homeTeamStartLeft`` from metadata JSON). Drives per-period
        direction-of-play normalization.
    home_team_start_left_extratime : bool or None, default None
        Same flag for ET periods 3/4 (``homeTeamStartLeftExtraTime`` from
        metadata JSON). Required only if the events span ET; raises
        ``ValueError`` if events have ``period_id`` ∈ {3, 4} but this is
        ``None``.
    preserve_native : list[str] or None, default None
        Optional input columns to passthrough into the output unchanged.

    Returns
    -------
    tuple[pd.DataFrame, ConversionReport]
        SPADL actions matching :data:`silly_kicks.spadl.GRADIENTSPORTS_SPADL_COLUMNS`
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

        from silly_kicks.spadl import gradientsports
        actions, report = gradientsports.convert_to_actions(
            events,
            home_team_id=366,             # Netherlands in WC 2022 NED-USA
            home_team_start_left=True,    # from match metadata
        )
        assert not report.has_unrecognized
    """
    _validate_input_columns(events, set(EXPECTED_INPUT_COLUMNS), provider="gradientsports")
    _validate_preserve_native(events, preserve_native, provider="gradientsports", schema=GRADIENTSPORTS_SPADL_COLUMNS)

    total_events_input = len(events)

    # PR-S23 / silly-kicks 3.0.1: validator re-enabled after TF-22 detector
    # hardening. Gradient Sports events ship PER_PERIOD_ABSOLUTE; the detector now
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
    direction.require_et_direction(
        events["period_id"], home_team_start_left_extratime, source="gradientsports convert_to_actions"
    )
    home_attacks_right_per_period = direction.home_attacks_right_per_period(
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

    # Component 4 (ADR-0NN): exclude voided ("annulled") events — possessionEvents.nonEvent == True
    # (play called back for a foul/advantage/offside; disallowed goals). Optional column: when absent,
    # an OBSERVABLE no-op (warn + omit the report key) so an under-equipped caller is not silently left
    # emitting voided events (incl. phantom goals) — the silent-undercount failure mode this guards.
    if "nonEvent" in events.columns:
        # Robust bool coercion (NOT .astype(bool) — that maps the string "false" to True and would
        # INVERT the exclusion, dropping real events and keeping voided ones). Only true-ish counts.
        _ne = events["nonEvent"]
        if _ne.dtype == bool:
            is_nonevent = _ne.fillna(False).to_numpy()
        else:

            def _truthy(v: object) -> bool:
                # Handles Python AND numpy bool (np.True_), strings, None/NaN. Avoids the `v is True`
                # trap (False for np.True_) and the `.astype(bool)` trap ("false" -> True).
                if isinstance(v, str):
                    return v.strip().lower() == "true"
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return False
                return bool(v)

            is_nonevent = _ne.map(_truthy).fillna(False).astype(bool).to_numpy()
        excluded_counts["nonEvent"] = int((is_nonevent & ~is_excluded).sum())
        is_excluded = is_excluded | is_nonevent
    else:
        warnings.warn(
            "gradientsports: 'nonEvent' column not supplied — voided events (annulled plays, "
            "including disallowed goals) are NOT excluded. Map possessionEvents.nonEvent into the "
            "converter input to enable Component-4 exclusion.",
            UserWarning,
            stacklevel=2,
        )
        # excluded_counts intentionally has NO 'nonEvent' key here: "not checked" != "0 voided".

    events = events.loc[~is_excluded].reset_index(drop=True)

    # Empty-input fast path (after exclusion): empty schema-compliant output.
    if len(events) == 0:
        actions = pd.DataFrame({col: [] for col in GRADIENTSPORTS_SPADL_COLUMNS.keys()})
        for col in (
            "tackle_winner_player_id",
            "tackle_winner_team_id",
            "tackle_loser_player_id",
            "tackle_loser_team_id",
        ):
            actions[col] = pd.array([], dtype="Int64")
        actions = _finalize_output(actions, schema=GRADIENTSPORTS_SPADL_COLUMNS)
        report = ConversionReport(
            provider="gradientsports",
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
    # Coordinate translation (centered → SPADL bottom-left meters)
    # ------------------------------------------------------------------
    actions = pd.DataFrame(
        {
            "game_id": events["game_id"].astype("int64").values,
            "original_event_id": events["event_id"].astype("object").values,
            "action_id": np.arange(len(events), dtype="int64"),
            "period_id": events["period_id"].astype("int64").values,
            "time_seconds": events["time_seconds"].astype("float64").values,
            # team_id / player_id mirror the canonical gameEvents actor as nullable
            # Int64 (NaN where the actor is absent — the null-actor duel/foul events).
            # NEVER a sentinel 0: see _resolve_team_ids + ADR-001. team_id self-heals
            # from the canonical player's same-match team where resolvable.
            "team_id": _resolve_team_ids(events),
            "player_id": events["player_id"].astype("Int64").reset_index(drop=True).array,
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
            # Provenance (ADR-018): True on converter-INJECTED rows (synthesized fouls + cross-goal
            # shots) that share their parent's original_event_id. Real 1:1 rows are False. Lets
            # consumers avoid collapsing/dropping a synthesized row when de-duping on original_event_id.
            "is_synthetic": np.zeros(len(events), dtype=bool),
        }
    )

    # ------------------------------------------------------------------
    # Impute NaN time_seconds via forward-fill + back-fill within period.
    # Real Gradient Sports data has NULL startGameClock on all dedicated
    # FOUL events (gameEventType=FOUL, possessionEventType=FO) — 28/28
    # across 13/64 WC2022 matches. Events are chronologically ordered
    # within a period, so ffill propagates the preceding event's timestamp
    # and bfill handles period-leading NaN.
    # ------------------------------------------------------------------
    if actions["time_seconds"].isna().any():
        actions["time_seconds"] = actions.groupby("period_id")["time_seconds"].transform(lambda s: s.ffill().bfill())

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
    # Derive end_x/end_y from next-action start for pass-class types.
    # Must run BEFORE foul synthesis: synthesized foul rows interleave
    # via 0.5-offset sort key and would intercept the shift(-1) chain.
    #
    # PR-S116: GS OTB/BC carries map to SPADL dribble with a placeholder end
    # (this converter initializes end=start for every event and never runs
    # _add_dribbles), so dribbles join the derive set GS-LOCALLY via
    # extra_type_ids. The shared _DERIVE_END_TYPE_IDS is unchanged on purpose:
    # its placeholder guard cannot distinguish statsbomb's ~11% genuine
    # stationary carries from placeholders.
    # ------------------------------------------------------------------
    actions = _derive_end_coordinates(actions, extra_type_ids=frozenset({spadlconfig.actiontype_id["dribble"]}))

    # ------------------------------------------------------------------
    # Foul row handling (two paths, depending on parent dispatch result)
    #
    # Gradient Sports places foul info in a per-event ``fouls`` dict that may co-occur with:
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

    # ---- Combined synthesis: foul rows (.5) + cross-goal shot rows (.4) -> one insert + renumber ----
    # Invariant (round-2 #4): the synthesis masks are computed on `events` and applied to `actions`
    # positionally, so they MUST still be 1:1 + index-aligned here (exclusion reset_index'd events;
    # actions built 1:1; _derive_end_coordinates is in-place). Fail loud if a future row-op breaks it.
    if len(actions) != len(events):  # internal invariant — actions built 1:1 with post-exclusion events
        raise RuntimeError(
            f"gradientsports synthesis precondition violated: {len(actions)} actions != {len(events)} events"
        )

    synth_parts: list[pd.DataFrame] = []
    _base_order = np.arange(len(actions), dtype="float64")
    actions["__order__"] = _base_order

    # Foul rows: parent already dispatched to a real action (synthesize an ADDITIONAL foul row).
    if synth_mask.any():
        foul_rows = actions.loc[synth_mask].copy()
        foul_rows["type_id"] = foul_id
        foul_rows["result_id"] = foul_result_full[synth_mask]
        foul_rows["bodypart_id"] = spadlconfig.bodypart_id["foot"]
        foul_rows["is_synthetic"] = True
        foul_rows["__order__"] = _base_order[synth_mask] + 0.5
        synth_parts.append(foul_rows)

    # Component 2 (ADR-0NN): cross-goal -> keep the cross, synthesize a shot by the crosser. SPADL
    # records a normal goal only as shot+success, so a direct cross-goal must register as a shot.
    # (`events` is still 1:1 with `actions`; recompute the CR mask from the post-exclusion events,
    # NOT the stale pre-exclusion pe_arr_full.)
    cg_mask = (events["possession_event_type"].fillna("").to_numpy() == "CR") & (
        events["shot_outcome_type"].fillna("").to_numpy() == "G"
    )
    if cg_mask.any():
        sp_cg = events["set_piece_type"].fillna("").to_numpy()[cg_mask]
        cg_type = np.select(
            [sp_cg == "F", sp_cg == "P"],
            [spadlconfig.actiontype_id["shot_freekick"], spadlconfig.actiontype_id["shot_penalty"]],
            default=spadlconfig.actiontype_id["shot"],
        ).astype("int64")
        shot_rows = actions.loc[cg_mask].copy()
        shot_rows["type_id"] = cg_type
        shot_rows["result_id"] = spadlconfig.result_id["success"]
        shot_rows["is_synthetic"] = True
        shot_rows["__order__"] = _base_order[cg_mask] + 0.4  # before a same-parent foul (.5)
        synth_parts.append(shot_rows)

    if synth_parts:
        actions = pd.concat([actions, *synth_parts], ignore_index=True)
        actions = actions.sort_values("__order__").reset_index(drop=True)
        actions["action_id"] = np.arange(len(actions), dtype="int64")
    actions = actions.drop(columns="__order__")

    # ------------------------------------------------------------------
    # Per-period direction-of-play normalisation. Routed through the canonical
    # to_spadl_ltr dispatcher per ADR-006 (silly-kicks 3.0.0); behaviour
    # preserved exactly because the dispatcher's PER_PERIOD_ABSOLUTE branch
    # uses the same home_attacks_right_per_period mapping computed
    # from metadata flags above.
    # ------------------------------------------------------------------
    actions = to_spadl_ltr(
        actions,
        input_convention=PER_PERIOD_ABSOLUTE,
        home_team_id=home_team_id,
        home_attacks_right_per_period=home_attacks_right_per_period,
    )

    # ------------------------------------------------------------------
    # Component 1 own-goal geometry tripwire (post-LTR). SPADL-LTR puts the acting team attacking
    # toward high-x, so a true own goal's ball sits in its OWN half (start_x < field_length/2). An
    # RE+G owngoal in the attacking half is a likely rebound-GOAL or feed anomaly -> WARN + revert to
    # the default RE handling (keeper_save/fail). Converts the n=3 rule into a self-policing one
    # (the owner-gated e2e validates the inequality on the 3 real WC2022 own goals). See ADR-0NN.
    # ------------------------------------------------------------------
    _og = (actions["result_id"] == spadlconfig.result_id["owngoal"]).to_numpy()
    if _og.any():
        _bad = _og & (actions["start_x"].to_numpy() >= spadlconfig.field_length / 2.0)
        if _bad.any():
            warnings.warn(
                f"gradientsports: {int(_bad.sum())} RE+G own-goal(s) with the ball in the acting "
                "team's attacking half (start_x >= field_length/2) — reverting to keeper_save/fail "
                "(likely a rebound-goal or feed anomaly, not an own goal).",
                UserWarning,
                stacklevel=2,
            )
            actions.loc[_bad, "type_id"] = spadlconfig.actiontype_id["keeper_save"]
            actions.loc[_bad, "result_id"] = spadlconfig.result_id["fail"]

    # ------------------------------------------------------------------
    # Clip coordinates to SPADL pitch bounds [0, 105] x [0, 68].
    # GS source tracking data reports positions slightly outside the
    # field lines (~1.2% of WC2022 actions, max ~5m x / ~8m y overshoot).
    # ------------------------------------------------------------------
    actions["start_x"] = actions["start_x"].clip(0, spadlconfig.field_length)
    actions["start_y"] = actions["start_y"].clip(0, spadlconfig.field_width)
    actions["end_x"] = actions["end_x"].clip(0, spadlconfig.field_length)
    actions["end_y"] = actions["end_y"].clip(0, spadlconfig.field_width)

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    actions = _finalize_output(
        actions,
        schema=GRADIENTSPORTS_SPADL_COLUMNS,
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
    # post-synthesis lookup -- gameEventId is not row-unique (multiple
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
        provider="gradientsports",
        total_events=total_events_input,
        total_actions=len(actions),
        mapped_counts=dict(mapped_counts),
        excluded_counts=dict(excluded_counts),
        unrecognized_counts=dict(unrecognized_counts),
    )
    return actions, report
