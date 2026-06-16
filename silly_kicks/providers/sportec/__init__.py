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
    "MatchInfo",
    "SportecEventBronze",
    "SportecTrackingBronze",
    "derive_idsse_home_team_start_left",
    "derive_idsse_home_team_start_left_extratime",
    "parse_dfl_events",
    "parse_dfl_match_info",
    "parse_dfl_tracking",
    "shape_events_to_native",
    "shape_tracking_to_native",
]
