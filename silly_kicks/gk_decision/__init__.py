"""silly-kicks GK build-up Decision-quality metric (TF-62): chosen-vs-available decision value.

Reframes xT-GK from possession VALUE (quarantined -- degenerate + ~80% team-confounded) to chosen-vs-
available DECISION quality: ``decision_value = EV(chosen) - E[EV(available options)]``, with option value
``EV = completion * (1 + max(0, opponents_bypassed))``. Scoring the keeper's choice against the option
set they could have played normalises out the team-created option set by construction.

Native tier: SkillCorner GI ``passing_option`` option sets (:class:`SkillCornerGIOptionSet`).
Reconstruction tiers (SB360 freeze-frame + full-tracking): :class:`ReconstructedOptionSet` with an
injected ``PassCompletionModel`` xPass, the reused ``compute_packing_metrics`` progression, and a
reachability filter -- every option scored in action-LTR.

Hexagonal: the native tier imports only ``silly_kicks.id_compat`` + pandas/numpy (the parsed GI option
rows are injected into ``SkillCornerGIOptionSet``, so it does not import
``silly_kicks.providers.skillcorner``); the reconstruction tier imports ``silly_kicks.tracking`` PUBLIC
seams (``compute_packing_metrics`` / ``action_ltr_goal_map`` / linking) + ``silly_kicks.keeper_identity``
+ ``silly_kicks.reflection`` -- NEVER a ``silly_kicks.tracking._*`` private. Nothing imports
``gk_decision``. Additive -- ``compute_*`` not ``add_*`` (no action-coupled aggregator), in NO default
xfn list, no VAEP/tracking retrain.

Deliverable = method + calibrated uncertainty + a multi-provider path -- NOT a keeper leaderboard (the
keeper-vs-team confound is unidentifiable without a multi-club transfer corpus; ranking is a future
ADR-009 gate). See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._columns import GK_DECISION_DROP_REASONS, OPTION_SET_SOURCE_VALUES
from ._compute import compute_gk_decision_value, summarize_gk_decision
from ._config import GkDecisionParams
from ._optionset import OptionSet, SkillCornerGIOptionSet
from ._reconstruct import ReconstructedOptionSet
from ._report import GkDecisionReport
from ._value import option_value

__all__ = [
    "GK_DECISION_DROP_REASONS",
    "OPTION_SET_SOURCE_VALUES",
    "GkDecisionParams",
    "GkDecisionReport",
    "OptionSet",
    "ReconstructedOptionSet",
    "SkillCornerGIOptionSet",
    "compute_gk_decision_value",
    "option_value",
    "summarize_gk_decision",
]
