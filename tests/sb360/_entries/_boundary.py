"""SB360 verdicts -- frame-consuming entry points OUTSIDE ``tracking.__all__``.

Observations are TRANSCRIBED FROM EXECUTION; only a human writes an adjudication or rationale.
"""

from __future__ import annotations

import silly_kicks.spadl as spadl
from tests.sb360._registry import AxisVerdict, _entry


def _call_restart_coordinates(actions, frames, links, home_team_id):
    """``add_restart_coordinates(actions, *, frames, links)`` -- no ``home_team_id``."""
    return spadl.add_restart_coordinates(actions, frames=frames, links=links)


_WORKS = "works"
_ALL = (
    "enriched_start_x",
    "enriched_start_y",
    "start_coord_source",
    "start_coord_confidence",
    "enriched_end_x",
    "enriched_end_y",
    "end_coord_source",
    "end_coord_confidence",
)

_entry(
    "spadl.add_restart_coordinates",
    _call_restart_coordinates,
    columns=_ALL,
    velocity={c: AxisVerdict("identical", _WORKS) for c in _ALL},
    visibility={
        "gk_absent": {c: AxisVerdict("identical", _WORKS) for c in _ALL},
        "defender_absent": {c: AxisVerdict("identical", _WORKS) for c in _ALL},
        # MEASURED, not assumed by symmetry: all 8 columns observed `identical` on gk_one_end,
        # same as the other two rosters. This entry is hand-maintained -- `_regenerate.py` loops
        # `tracking.__all__` and this is a BOUNDARY_ENTRY_POINT outside it -- so a new roster does
        # NOT reach it automatically, and `test_every_visibility_roster_has_its_own_slot` is what
        # says so (it failed with a KeyError until this block existed).
        "gk_one_end": {c: AxisVerdict("identical", _WORKS) for c in _ALL},
    },
    # ADR-025 imputes restart coordinates from Law-fixed spots and the action's own geometry;
    # nothing here reads another player's position, so both probes correctly move nothing.
    applicability={c: "no_support" for c in _ALL},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _ALL},
)
