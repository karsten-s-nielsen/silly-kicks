"""silly-kicks GK shot-stopping metrics -- Goals Prevented / GSAA (TF-59 PR2).

Event-only, per-(goalkeeper, match): from SPADL ``actions`` + an INJECTED per-shot Post-Shot xG
(``psxg_column``; silly-kicks ships no xG model) + the PR1-stamped ``defending_gk_player_id`` /
``defending_gk_team_id`` columns, compute Goals Prevented (== GSAA = sum(PSxG faced) - goals conceded),
reported with and without in-play penalties. Own goals / blocked shots / the penalty shootout are
excluded (spec §6.2).

Hexagonal / event-only: imports ``silly_kicks.spadl`` (config) + ``silly_kicks.id_compat`` +
``silly_kicks.keeper_identity`` ONLY; NEVER ``silly_kicks.tracking`` (pinned by
``tests/shot_stopping/test_import_allowlist.py``). NOTHING imports ``shot_stopping``. Additive -- no
VAEP/tracking retrain.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._columns import SHOT_STOPPING_COLUMNS, SHOT_STOPPING_METRIC_COLUMNS, SS_KEYS
from ._compute import compute_shot_stopping
from ._config import ShotStoppingParams
from ._report import ShotStoppingReport

__all__ = [
    "SHOT_STOPPING_COLUMNS",
    "SHOT_STOPPING_METRIC_COLUMNS",
    "SS_KEYS",
    "ShotStoppingParams",
    "ShotStoppingReport",
    "compute_shot_stopping",
]
