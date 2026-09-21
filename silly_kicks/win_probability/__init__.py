"""TF-63 in-game win-probability + xImpact.

A self-contained, event-only package: an interval-hazard GLM feeds a forward Markov chain on
``score_diff`` (it replaces, not reuses, the TF-53 convolution/simplex). ``goal_leverage`` reads the
exact per-state ``ΔP(win | goal)`` from the chain; ``VAEP.rate_ximpact`` (in ``vaep``) multiplies it by
``VAEP_adjusted``. Imports no ``tracking``, no ``vaep``, no ``match_outcome``.

See ``docs/superpowers/specs/2026-09-20-tf63-ximpact-ingame-winprob-design.md``.
"""

from __future__ import annotations

from ._columns import WIN_PROBABILITY_COLUMNS, WIN_PROBABILITY_KEYS
from ._compute import compute_win_probability, goal_leverage
from ._config import WinProbabilityParams
from ._model import WinProbabilityIntegrityError, WinProbabilityModel
from ._report import WinProbabilityReport

__all__ = [
    "WIN_PROBABILITY_COLUMNS",
    "WIN_PROBABILITY_KEYS",
    "WinProbabilityIntegrityError",
    "WinProbabilityModel",
    "WinProbabilityParams",
    "WinProbabilityReport",
    "compute_win_probability",
    "goal_leverage",
]
