# ruff: noqa: F405
"""Public-API package for ``silly_kicks.vaep.features``.

Decomposed in 2.3.0 from a 1170-line monolith into 8 concern-focused submodules.
Submodules are importable directly (``silly_kicks.vaep.features.spatial.startlocation``),
but the canonical entry point remains the package itself
(``silly_kicks.vaep.features.startlocation``). All 33 previously-public symbols
remain importable via the package path; new code is encouraged to use the
package import for backwards-compat.

See ``docs/superpowers/specs/2026-04-30-vaep-features-decomposition-design.md``
for the rationale and submodule layout.
"""

from silly_kicks.vaep.feature_framework import *  # noqa: F403

from .actiontype import *  # noqa: F403
from .bodypart import *  # noqa: F403
from .context import *  # noqa: F403
from .core import *  # noqa: F403
from .result import *  # noqa: F403
from .spatial import *  # noqa: F403
from .specialty import *  # noqa: F403
from .temporal import *  # noqa: F403

# Static __all__: union of every submodule's __all__, listed verbatim for
# pyright friendliness (avoids reportUnsupportedDunderAll on dynamic
# union-via-spread). Keep alphabetised. When a submodule adds a public name,
# add the corresponding entry here.
__all__ = [
    "Actions",
    "FeatureTransfomer",
    "Features",
    "GameStates",
    "actiontype",
    "actiontype_categorical",
    "actiontype_onehot",
    "actiontype_result_onehot",
    "actiontype_result_onehot_prev_only",
    "assist_type",
    "bodypart",
    "bodypart_detailed",
    "bodypart_detailed_onehot",
    "bodypart_onehot",
    "cross_zone",
    "endlocation",
    "endpolar",
    "feature_column_names",
    "gamestates",
    "goalscore",
    "movement",
    "play_left_to_right",
    "player_possession_time",
    "result",
    "result_onehot",
    "result_onehot_prev_only",
    "simple",
    "space_delta",
    "speed",
    "startlocation",
    "startpolar",
    "team",
    "time",
    "time_delta",
]
