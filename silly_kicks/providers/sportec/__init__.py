"""DFL / Sportec parse+shape port (ADR-031 T3).

Public surface:

* ``parse_dfl_match_info`` / ``parse_dfl_tracking`` / ``parse_dfl_events`` --- the
  faithful ``bytes -> bronze`` parse layer (DFL XML -> provider-canonical bronze
  rows), upstreamed verbatim from ``luxury-lakehouse`` ``src/ingestion/idsse.py``.
* ``shape_tracking_to_native`` / ``shape_events_to_native`` --- the
  ``bronze -> silly-kicks-converter-input`` shape layer.
* ``MatchInfo`` / ``SportecTrackingBronze`` / ``SportecEventBronze`` --- the typed
  returns (silly-kicks' own domain names; field-identical to the lakehouse bronze
  tables today --- a versioned cross-repo contract, ADR-031 N1).

See ``docs/superpowers/specs/2026-06-16-dfl-parse-port-design.md``.
"""

from __future__ import annotations

from ._event_derivation import (
    DFL_EVENT_TYPE_PREFIX,
    DFL_NESTED_PREFIX,
    DflEventTimeWarning,
    DflGameSectionWarning,
    DflPeriodReport,
    complete_dfl_event_bronze,
    derive_dfl_event_periods,
    dfl_actor_columns,
    dfl_event_column,
    resolve_dfl_event_actors,
)
from .appearances import extract_keeper_appearances
from .parse import (
    MatchInfo,
    SportecEventBronze,
    SportecTrackingBronze,
    derive_idsse_home_team_start_left,
    derive_idsse_home_team_start_left_extratime,
    parse_dfl_events,
    parse_dfl_match_info,
    parse_dfl_tracking,
    shape_events_to_native,
    shape_tracking_to_native,
)

__all__ = [
    "DFL_EVENT_TYPE_PREFIX",
    "DFL_NESTED_PREFIX",
    "DflEventTimeWarning",
    "DflGameSectionWarning",
    "DflPeriodReport",
    "MatchInfo",
    "SportecEventBronze",
    "SportecTrackingBronze",
    "complete_dfl_event_bronze",
    "derive_dfl_event_periods",
    "derive_idsse_home_team_start_left",
    "derive_idsse_home_team_start_left_extratime",
    "dfl_actor_columns",
    "dfl_event_column",
    "extract_keeper_appearances",
    "parse_dfl_events",
    "parse_dfl_match_info",
    "parse_dfl_tracking",
    "resolve_dfl_event_actors",
    "shape_events_to_native",
    "shape_tracking_to_native",
]
