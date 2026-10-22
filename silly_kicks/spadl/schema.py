"""SPADL output schema — plain Python constants.

These constants define the guaranteed output contract of convert_to_actions().
They replace the pandera DataFrameModel that previously served this role.
"""

import dataclasses

SPADL_COLUMNS: dict[str, str] = {
    "game_id": "int64",
    "original_event_id": "object",
    "action_id": "int64",
    "period_id": "int64",
    # PERIOD-RELATIVE: seconds since the start of the period, resets to 0 each period (ADR-017)
    "time_seconds": "float64",
    "team_id": "int64",
    "player_id": "int64",
    "start_x": "float64",
    "start_y": "float64",
    "end_x": "float64",
    "end_y": "float64",
    "type_id": "int64",
    "result_id": "int64",
    "bodypart_id": "int64",
    # Block-detection columns (TF-51 prereq). Nullable "boolean": True/False on shot/cross rows
    # where the provider encodes a block, pd.NA on non-shot/non-cross rows AND on providers that
    # cannot encode the signal (Opta, SkillCorner). See the block-detection spec.
    "shot_blocked": "boolean",
    "cross_blocked": "boolean",
    # TF-46 Arm 1 base signal columns (ADR-046 pattern: a cross-provider signal read by a shared
    # consumer lives on the base schema, nullable, pd.NA where a provider or row does not carry it).
    # The shared training-exclusion helper reads result_source + restart_evidence; the corner layer
    # reads type_source. Nullable string ("object"). See the TF-46 Arm 1 corrections spec (decision D).
    "result_source": "object",  # native / inferred / stopgap; <NA> = tier not classified
    "restart_evidence": "object",  # tagged / untagged / chain / chain_ambiguous / tag_only; <NA> otherwise
    "type_source": "object",  # native / inferred / default / tracking; <NA> where no subtype judgement
    # repair_kind is a base signal column (TF-46 Arm 1 Part 2 decision D1): two emitters (SkillCorner
    # goals-reconciliation, Sportec §25 penalties) and shared readers (training_exclusion_mask,
    # xtgk markov / turnover) meet the Part-1 decision-D base-promotion rule. <NA> where not carried.
    "repair_kind": "object",  # a goals-reconciliation / penalty-outcome provenance class; <NA> on a clean row
}

#: The closed vocabulary of ``repair_kind`` (TF-46 Arm 1 Part 2 decision D1). Every emitted value is in
#: this set (asserted by ``tests/spadl/test_schema.py``). SkillCorner goals-reconciliation (Part 1) emits
#: the first group; Sportec §25 penalties emit the last two.
REPAIR_KIND_VALUES: frozenset[str] = frozenset(
    {
        "normal",
        "scored_tag",
        "own_goal_inverted",
        "own_goal_surplus",
        "own_goal_unsignalled",
        "reading_i",
        "missing_goal",
        "penalty_outcome_unresolved",
        "penalty_score_inconsistent",
    }
)

SPADL_NAME_COLUMNS: dict[str, str] = {
    "type_name": "object",
    "result_name": "object",
    "bodypart_name": "object",
}

SPADL_CONSTRAINTS: dict[str, tuple[float, float]] = {
    "period_id": (1, 5),
    "time_seconds": (0, float("inf")),
    "start_x": (0, 105.0),
    "start_y": (0, 68.0),
    "end_x": (0, 105.0),
    "end_y": (0, 68.0),
}

KLOPPY_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    "game_id": "object",
    "team_id": "object",
    "player_id": "object",
}

SPORTEC_SPADL_COLUMNS: dict[str, str] = {
    **KLOPPY_SPADL_COLUMNS,
    "tackle_winner_player_id": "object",
    "tackle_winner_team_id": "object",
    "tackle_loser_player_id": "object",
    "tackle_loser_team_id": "object",
    "action_provenance": "object",
    "ball_claim_type": "object",
    # TF-46 Arm 1 Part 2 §25 (penalty mapping / BallDeflection routing). repair_kind is NOT here — it is a
    # base SPADL_COLUMNS signal column (decision D1), inherited through KLOPPY_SPADL_COLUMNS.
    "penalty_result_method": "object",  # shot_outcome / score_reconciliation / kickoff_order /
    #                                     continued_play / score_elimination / unresolved; <NA> off penalty rows
    "shot_woodwork_type": "object",  # post_left / post_right / crossbar; <NA> off woodwork-deflected shots
}
"""Sportec SPADL output schema: KLOPPY_SPADL_COLUMNS + 8 sportec-specific
qualifier passthrough / provenance columns (the TF-46 Arm 1 Part 2 §25 additions
``penalty_result_method`` and ``shot_woodwork_type`` join the earlier six; ``repair_kind`` is a base
SPADL_COLUMNS signal column, decision D1, not sportec-specific).

The 4 tackle columns surface the DFL ``tackle_winner`` / ``tackle_winner_team`` /
``tackle_loser`` / ``tackle_loser_team`` qualifier values verbatim. NaN on rows
where the qualifier is absent in the source; always NaN on non-tackle rows. See
ADR-001 for the dedicated-column contract rationale.

``action_provenance`` (``"native"`` / ``"derived"``) marks rows the converter
synthesizes rather than lifts one-to-one from a DFL event: the F4 ``BallClaiming``
interception / keeper_pick_up rows and the ``throwOut`` / ``punt`` keeper_pick_up
distribution halves. ``native`` elsewhere.

``ball_claim_type`` carries the DFL ``BallClaiming`` ``claim_type``
(``InterceptedBall`` / ``BallClaimed`` / ``BallHeld`` / ``Block``) on the partner
action the claim annotates (ADR-001 dedicated column); NaN where the row is not a
``BallClaiming`` partner. See the TF-46 Arm 1 corrections spec (F4)."""

GRADIENTSPORTS_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    # team_id / player_id are nullable Int64 (NOT the SPADL_COLUMNS int64): the
    # canonical gameEvents actor is genuinely absent on the null-actor duel/foul
    # events (OTB+CH challenges, FOUL+FO fouls), so team_id/player_id must carry
    # NaN there rather than a sentinel 0 that masquerades as a real id and
    # crashes downstream id-resolution guards. ADR-001 forbids back-filling them
    # from the duel/foul qualifiers; NaN is the correct "no single actor" value.
    "team_id": "Int64",
    "player_id": "Int64",
    "tackle_winner_player_id": "Int64",
    "tackle_winner_team_id": "Int64",
    "tackle_loser_player_id": "Int64",
    "tackle_loser_team_id": "Int64",
    "is_synthetic": "bool",
}
"""Gradient Sports (formerly PFF FC) SPADL output schema: SPADL_COLUMNS with
nullable-``Int64`` ``team_id`` / ``player_id`` + 4 nullable Int64 tackle-actor
passthrough columns. NaN on rows where no challenge winner/loser is
identifiable (i.e., everywhere except CH events), and NaN on ``team_id`` /
``player_id`` for the null-actor duel/foul events.

Identifier-conventions rationale (ADR-001) shared with SPORTEC_SPADL_COLUMNS.

Dtype departure from SPADL_COLUMNS (which uses non-nullable ``int64`` for
``team_id`` / ``player_id``) AND from SPORTEC_SPADL_COLUMNS (which uses
``object`` strings): Gradient Sports native player/team identifiers are
integers, whereas kloppy hands sportec strings. Using ``Int64`` (pandas
nullable) preserves int-ness while allowing NaN where the canonical actor is
absent. Long-term unification of the two extended schemas under a common name
is a follow-up TODO."""


SKILLCORNER_SPADL_COLUMNS: dict[str, str] = {
    **KLOPPY_SPADL_COLUMNS,
    "action_provenance": "object",
    # result_source AND repair_kind now come from base SPADL_COLUMNS (ADR-046 / decision D / D1); not
    # re-declared here.
}
"""SkillCorner SPADL output schema: KLOPPY_SPADL_COLUMNS (object-dtype IDs) +
``action_provenance`` (``"native"`` or ``"derived"``). ``repair_kind`` (goals-reconciliation
provenance: ``owngoal`` / ``missing_goal`` / ``own_goal_unsignalled`` / ...; ``<NA>`` on un-repaired
rows) is now a base ``SPADL_COLUMNS`` column (decision D1), not re-declared here.
``result_source`` (``"native"`` / ``"inferred"`` / ``"stopgap"``: the completion-label tier
behind ``result_id``; native = SkillCorner ``pass_outcome``, inferred = ``received==True``, stopgap =
``same_team_next`` residual; D-S8/G1: only ``native`` trains the GK-completion model) now comes from
base ``SPADL_COLUMNS`` (ADR-046 / TF-46 Arm 1 decision D), not re-declared here. Derived actions include
``start_type``-based interceptions/recoveries, OBE-enriched tackles, keeper saves, and synthetic
dribbles. See the TF-46 Arm 1 corrections spec."""


@dataclasses.dataclass(frozen=True)
class ConversionReport:
    """Audit trail for convert_to_actions().

    Every converter returns this alongside the actions DataFrame, enabling
    callers to detect silent event drops and new event types.

    Attributes:
        provider: Name of the data provider (e.g. "StatsBomb", "Wyscout").
        total_events: Number of input events before conversion.
        total_actions: Number of output SPADL actions produced.
        mapped_counts: Per-type counts of events successfully mapped to SPADL.
        excluded_counts: Per-type counts of events intentionally excluded
            (e.g. "Half Start", "Referee Ball-Drop").
        unrecognized_counts: Per-type counts of events not in the mapped or
            excluded registries.  Keys are provider-specific: ``str`` for
            StatsBomb (event type names), ``int`` for Wyscout (type IDs).

    Example::

        actions, report = statsbomb.convert_to_actions(events, home_team_id=100)
        if report.has_unrecognized:
            logger.warning("Unrecognized events: %s", report.unrecognized_counts)
        logger.info("Converted %d events -> %d actions", report.total_events, report.total_actions)
    """

    provider: str
    total_events: int
    total_actions: int
    mapped_counts: dict[str, int]
    excluded_counts: dict[str, int]
    unrecognized_counts: dict[str, int]

    @property
    def has_unrecognized(self) -> bool:
        """Return True if any unrecognized event types were encountered."""
        return len(self.unrecognized_counts) > 0


@dataclasses.dataclass(frozen=True)
class SkillCornerConversionReport(ConversionReport):
    """SkillCorner conversion report: the base audit plus the TF-46 restart + goals diagnostics.

    An ``isinstance(report, ConversionReport)`` check still holds. The appended fields carry the derived
    restart census (per type and per ``restart_evidence`` class), the goals-reconciliation summary, and
    the C4 repeated-restart / kick-off diagnostics. ``dataclasses.asdict`` round-trips every field.

    Examples
    --------
    Read the restart + goals diagnostics off a SkillCorner conversion::

        actions, report = skillcorner.convert_to_actions(events, match_metadata)
        report.restart_counts          # e.g. {"corner_crossed": 9, "throw_in": 44, ...}
        report.restart_evidence_counts # e.g. {"tagged": 79, "chain_ambiguous": 2, ...}
        report.goals_repaired          # e.g. {"normal": 2, "own_goal_unsignalled": 1}
        assert not report.goals_coverage_gap
    """

    restart_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    restart_evidence_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    restart_unresolved: int = 0
    goals_repaired: dict[str, int] = dataclasses.field(default_factory=dict)
    goals_ambiguous: int = 0
    goals_unresolved: int = 0
    goals_coverage_gap: bool = False
    repeated_restart_count: int = 0
    kickoff_unresolved: int = 0


@dataclasses.dataclass(frozen=True)
class SportecConversionReport(ConversionReport):
    """Sportec conversion report: the base audit plus the TF-46 Arm 1 F4 / F7 diagnostics.

    An ``isinstance(report, ConversionReport)`` check still holds. ``dataclasses.asdict`` round-trips every
    field. The appended fields carry the ``BallClaiming`` (F4) folding census and the free-kick-crossed
    (F7) stopgap count:

    - ``ball_claim_counts`` -- ``BallClaiming`` events seen, keyed by ``claim_type``
      (``InterceptedBall`` / ``BallClaimed`` / ``BallHeld`` / ``Block``).
    - ``interceptions_added`` / ``keeper_pick_ups_added`` -- derived rows emitted by F4.
    - ``ball_claim_unattached`` -- qualifier-only claims (``BallClaimed`` / ``Block``) whose partner action
      did not survive, so the qualifier had nowhere to land.
    - ``freekick_inferred_count`` -- free kicks that fell back to ``freekick_short`` with
      ``type_source="inferred"`` because the DFL ``play_penalty_box`` qualifier was absent.

    Examples
    --------
    Read the F4 / F7 diagnostics off a Sportec conversion::

        actions, report = sportec.convert_to_actions(events, home_team_id="DFL-CLU-...")
        report.ball_claim_counts        # e.g. {"InterceptedBall": 146, "BallClaimed": 42, ...}
        report.interceptions_added      # e.g. 134
        report.freekick_inferred_count  # e.g. 4
    """

    ball_claim_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    interceptions_added: int = 0
    keeper_pick_ups_added: int = 0
    ball_claim_unattached: int = 0
    freekick_inferred_count: int = 0
    # TF-46 Arm 1 Part 2 §25: DFL Penalty result-method census and dropped-retake count, plus the
    # BallDeflection routing census (woodwork_paired / deflection_referee / deflection_unpaired).
    penalty_result_method_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    penalty_retakes: int = 0
    deflection_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    own_goal_attribution_conflicts: int = 0
    # C-16: teams whose opponent was unresolvable during penalty reconciliation (0 on a well-formed two-team
    # feed). A visibility counter for a degenerate frame where steps 3/3m emit no opponent-dependent result.
    penalty_opponent_unresolved: int = 0
