"""Implements the Hybrid-VAEP framework.

Hybrid-VAEP removes result leakage from the current action's features.
Standard VAEP includes the action's result (success/fail) as a feature,
which means the model knows the outcome before valuing the action. This
undercredits defenders and pass receivers.

HybridVAEP uses result_onehot_prev_only and actiontype_result_onehot_prev_only
to include result information only for previous actions (a1, a2) where the
result is already known, not for the current action (a0).
"""

from collections.abc import Callable

from silly_kicks.vaep.base import VAEP

from . import features as fs

hybrid_xfns_default = [
    fs.actiontype_onehot,
    fs.result_onehot_prev_only,
    fs.actiontype_result_onehot_prev_only,
    fs.bodypart_onehot,
    fs.time,
    fs.startlocation,
    fs.endlocation,
    fs.startpolar,
    fs.endpolar,
    fs.movement,
    fs.team,
    fs.time_delta,
    fs.space_delta,
    fs.goalscore,
]


class HybridVAEP(VAEP):
    """VAEP with result leakage removed from current-action features.

    In standard VAEP, the model receives the action's result (success/fail)
    as a feature, which creates information leakage. HybridVAEP removes
    result information from the current action (a0) while preserving it
    for previous actions (a1, a2) where the result is already known.

    Parameters
    ----------
    xfns : list, optional
        Feature transformers. Uses hybrid_xfns_default if None.
    yfns : list, optional
        Label functions. Uses [scores, concedes] if None.
    nb_prev_actions : int, default=3
        Number of previous actions in game state.
    """

    def __init__(
        self,
        xfns: list[fs.FeatureTransfomer] | None = None,
        yfns: list[Callable] | None = None,
        nb_prev_actions: int = 3,
    ) -> None:
        xfns = list(hybrid_xfns_default) if xfns is None else xfns
        super().__init__(xfns, yfns, nb_prev_actions)
