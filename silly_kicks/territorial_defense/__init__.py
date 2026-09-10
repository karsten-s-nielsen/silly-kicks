"""silly_kicks.territorial_defense -- TF-54b SB360 territorial-defense counterfactual.

How much a defender's POSITIONING suppresses the attacking team's threat, via a model-free removal
(marginal-contribution) counterfactual on SB360 freeze-frames. Two arms: Arm A (action-anchored,
identity-exact) and Arm B (hull-based, attribution-approximate). Tracking-consuming sibling of
``gkdv`` / ``restdefense`` -- imports ``silly_kicks.tracking`` / ``keeper_identity`` / ``territory``
PUBLIC seams only; NOTHING imports it.

HONEST LIMIT -- validated as an INSTRUMENT, NOT player-attributable: the marginal-removal delta is
team-conditioned by construction, and on a single-tournament / national-team corpus the
defender-vs-team confound is unidentifiable, so per-defender numbers are NOT a defender ranking.

See NOTICE for full bibliographic citations.
"""

from ._columns import TD_SAMPLE_COLUMNS, TD_SAMPLE_KEYS, TD_SOURCE_VALUES
from ._compute import compute_territorial_defense
from ._config import TerritorialDefenseParams
from ._report import TerritorialDefenseReport

__all__ = [
    "TD_SAMPLE_COLUMNS",
    "TD_SAMPLE_KEYS",
    "TD_SOURCE_VALUES",
    "TerritorialDefenseParams",
    "TerritorialDefenseReport",
    "compute_territorial_defense",
]
