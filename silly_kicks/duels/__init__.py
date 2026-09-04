"""silly-kicks Glicko-2 duel ratings (TF-55).

Event-only, per-(player, match): a pairwise skill rating (rating + deviation + volatility) over duel
win/loss outcomes, updated per match (rating period). Native winner/loser on sportec
(``tackle_winner/loser``); derived from the ``tackle`` / ``take_on`` result adjacency elsewhere. Ground
duels only (no SPADL aerial type); indeterminate duels excluded. A pure ``update_glicko`` primitive +
a resumable ``compute_duel_ratings`` orchestrator.

Hexagonal / event-only: imports ``silly_kicks.spadl`` + ``silly_kicks.id_compat`` ONLY; NEVER
``silly_kicks.tracking`` (pinned by ``tests/duels/test_import_allowlist.py``). Additive -- no
VAEP/tracking retrain.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._columns import DUEL_COLUMNS, DUEL_METRIC_COLUMNS, DUEL_WINNER_SOURCE_VALUES
from ._compute import compute_duel_ratings, update_glicko
from ._config import DuelRatingParams, GlickoState
from ._extract import DuelExtractReport, DuelGame, extract_duels
from ._report import DuelRatingReport

__all__ = [
    "DUEL_COLUMNS",
    "DUEL_METRIC_COLUMNS",
    "DUEL_WINNER_SOURCE_VALUES",
    "DuelExtractReport",
    "DuelGame",
    "DuelRatingParams",
    "DuelRatingReport",
    "GlickoState",
    "compute_duel_ratings",
    "extract_duels",
    "update_glicko",
]
