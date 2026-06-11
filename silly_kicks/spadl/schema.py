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
}

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
}
"""Sportec SPADL output schema: KLOPPY_SPADL_COLUMNS + 4 sportec-specific
qualifier passthrough columns surfacing DFL ``tackle_winner`` /
``tackle_winner_team`` / ``tackle_loser`` / ``tackle_loser_team`` qualifier
values verbatim. NaN on rows where the qualifier is absent in the source;
always NaN on non-tackle rows. See ADR-001 for the contract rationale."""

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
    "result_source": "object",
}
"""SkillCorner SPADL output schema: KLOPPY_SPADL_COLUMNS (object-dtype IDs) +
``action_provenance`` column (``"native"`` or ``"derived"``) + ``result_source``
(``"native"`` / ``"inferred"`` / ``"stopgap"``: the pass-completion-label tier behind
``result_id``; native = SkillCorner ``pass_outcome``, inferred = ``received==True``,
stopgap = ``same_team_next`` residual. D-S8/G1: only ``native`` trains the GK-completion
model). Derived actions include ``start_type``-based interceptions/recoveries, OBE-enriched
tackles, keeper saves, and synthetic dribbles. See spec §6."""


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
