"""StatsBomb 360 parse port -- freeze-frames to the tracking-snapshot contract.

Shape only, never fetch: the caller owns I/O, matching ``providers/sportec/parse.py``. No new
runtime dependency -- ``statsbombpy`` is a script dependency and is not imported here.

See NOTICE for the StatsBomb Public Data License (non-commercial).
"""

from __future__ import annotations

from .parse import (
    ACTING_TEAM_ID,
    OPPONENT_TEAM_ID,
    JoinReport,
    acting_side_gk_visible,
    defending_gk_visible,
    polygon_to_spadl,
    shape_snapshots,
    visible_fraction,
)

__all__ = [
    "ACTING_TEAM_ID",
    "OPPONENT_TEAM_ID",
    "JoinReport",
    "acting_side_gk_visible",
    "defending_gk_visible",
    "polygon_to_spadl",
    "shape_snapshots",
    "visible_fraction",
]
