"""SPADL output schema — plain Python constants.

These constants define the guaranteed output contract of convert_to_actions().
They replace the pandera DataFrameModel that previously served this role.
"""

import dataclasses

import numpy as np
import pandas as pd

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
    # The provider's own "crossed" label (ADR-133). The open-play SPADL type is set by one geometric rule for every
    # provider (spadl.base.classify_deliveries); this keeps the native signal. True/False on open-play
    # pass-class rows and on corner/free-kick rows where the provider labels deliveries (there it equals the
    # type, which the rule does not re-type), pd.NA elsewhere and on providers with no native label (SkillCorner).
    "cross_native": "boolean",
    # The source of each row's time_seconds (action-time repair spec section 4.1): one of TIME_SOURCE_VALUES.
    # "event" = the provider's event clock; "tracking" = the provider's time on its own tracking clock;
    # "elastic" = set from the tracking by ELASTIC; "kick" = a restart or penalty kick measured from the
    # tracking; "placeholder" = a restart with a first possession, no measured kick (an upper bound of the
    # kick); "order_only" = a restart with no first possession (the time keeps the row order only);
    # "derived" = computed from the neighbouring rows (synthetic dribbles).
    "time_source": "object",
}

TIME_SOURCE_VALUES: tuple[str, ...] = ("event", "tracking", "elastic", "kick", "placeholder", "order_only", "derived")
"""The vocabulary of the ``time_source`` column; ``_finalize_output`` refuses any other value."""

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


def is_order_only(actions: pd.DataFrame) -> np.ndarray:
    """True where ``time_source == "order_only"``: a time that keeps the row order only (spec section 4.1).

    A reader that takes a row time as a moment, or as the bound of a time window, takes no window at such a row.
    All ``False`` when the table has no ``time_source`` column (a reader treats each row as ``"event"``).

    Examples
    --------
    >>> import pandas as pd
    >>> is_order_only(pd.DataFrame({"time_source": ["placeholder", "order_only", "event"]})).tolist()
    [False, True, False]
    """
    if "time_source" not in actions.columns:
        return np.zeros(len(actions), dtype=bool)
    return (actions["time_source"] == "order_only").to_numpy(dtype=bool)


def is_time_unmeasured(actions: pd.DataFrame) -> np.ndarray:
    """True where ``time_source`` is ``"placeholder"`` or ``"order_only"``: neither is the moment of the action.

    The readers of a shot row's time skip such a row (spec section 5.4). All ``False`` without the column.

    Examples
    --------
    >>> import pandas as pd
    >>> is_time_unmeasured(pd.DataFrame({"time_source": ["placeholder", "order_only", "kick"]})).tolist()
    [True, True, False]
    """
    if "time_source" not in actions.columns:
        return np.zeros(len(actions), dtype=bool)
    return actions["time_source"].isin(("placeholder", "order_only")).to_numpy(dtype=bool)


_RESTART_KINDS: dict[str, str] = {
    "corner_crossed": "corner",
    "corner_short": "corner",
    "freekick_crossed": "free_kick",
    "freekick_short": "free_kick",
    "goalkick": "goal_kick",
    "throw_in": "throw_in",
    "shot_penalty": "penalty",
}


def restart_kind(type_id: int) -> str:
    """The restart kind of a SPADL type id; the type name for any other type (spec section 4.18).

    The kind is part of the key of a SkillCorner row, so a step that re-sets a corner subtype leaves the key as it
    was.

    Examples
    --------
    >>> from silly_kicks.spadl import config as spadlconfig
    >>> restart_kind(spadlconfig.actiontype_id["corner_short"])
    'corner'
    >>> restart_kind(spadlconfig.actiontype_id["pass"])
    'pass'
    """
    from . import config as spadlconfig

    name = spadlconfig.actiontypes[int(type_id)]
    return _RESTART_KINDS.get(name, name)
