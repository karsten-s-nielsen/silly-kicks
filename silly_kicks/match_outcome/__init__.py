"""TF-53 match-outcome simulation -- win probability / xPoints from injected per-shot xG.

Event-only ``compute_*`` sibling (territory / duels / shot_stopping / gk_decision). Exact
Poisson-binomial core with two orthogonal opt-in honesty corrections (possession-collapse +
Dixon-Coles dependence). See ``docs/superpowers/specs/2026-09-16-tf53-match-outcome-design.md``.
"""

from __future__ import annotations

from ._columns import MATCH_OUTCOME_COLUMNS, MATCH_OUTCOME_KEYS, MATCH_OUTCOME_METRIC_COLUMNS
from ._compute import compute_match_outcome
from ._config import MatchOutcomeParams
from ._dependence import DependenceModel, MatchOutcomeIntegrityError, apply_dependence, dixon_coles_tau
from ._pmf import goal_count_pmf, match_outcome_probabilities
from ._report import MatchOutcomeReport

__all__ = [
    "MATCH_OUTCOME_COLUMNS",
    "MATCH_OUTCOME_KEYS",
    "MATCH_OUTCOME_METRIC_COLUMNS",
    "DependenceModel",
    "MatchOutcomeIntegrityError",
    "MatchOutcomeParams",
    "MatchOutcomeReport",
    "apply_dependence",
    "compute_match_outcome",
    "dixon_coles_tau",
    "goal_count_pmf",
    "match_outcome_probabilities",
]
