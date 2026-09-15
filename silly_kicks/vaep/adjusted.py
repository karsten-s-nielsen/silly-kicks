"""VAEP_adjusted (TF-61) — the risk-aware, outcome-bias-free VAEP combiner (Paul/Klemp/Memmert 2025).

Eqs 8-12: weight the counterfactual scoring/conceding probabilities by the action-completion
probability, then run the *existing* :func:`silly_kicks.vaep.formula.value` before/after delta
machinery. Reused verbatim so there is no new delta logic:

- ``p_scores_adj  = xSuccess * P(scores | success)``   (eq 8)
- ``p_concedes_adj = (1 - xSuccess) * P(concedes | fail)`` (eq 9)
- ``V_adj = ΔP(scores|success)_adj - ΔP(concedes|fail)_adj`` (eqs 10-12) == ``formula.value(...)``.

``VAEP.rate_adjusted`` (see ``vaep/base.py``) builds the surgical result-feature counterfactual,
re-scores the fitted classifiers, and calls this helper. See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import formula


def adjusted_value(actions: pd.DataFrame, p_scores_success, p_concedes_fail, xsuccess) -> pd.DataFrame:
    """Combine per-action completion probability with the counterfactual score/concede probabilities.

    Parameters
    ----------
    actions : pandas.DataFrame
        SPADL actions with ``add_names`` applied (``formula.value`` reads ``team_id`` /
        ``time_seconds`` / ``type_name`` / ``result_name``).
    p_scores_success : array-like
        ``P(scores | this action succeeds)`` per action (from the fitted P_scores classifier scored
        on the all-success result-feature counterfactual).
    p_concedes_fail : array-like
        ``P(concedes | this action fails)`` per action (fitted P_concedes on the all-fail counterfactual).
    xsuccess : array-like
        ``P(success)`` per action. A NaN entry propagates to a NaN adjusted value (never fabricated).

    Returns
    -------
    pandas.DataFrame
        ``offensive_value`` / ``defensive_value`` / ``vaep_value`` (adjusted), one row per action.

    Examples
    --------
    Combine the counterfactual probabilities with per-action completion (a real ``actions`` frame with
    ``add_names`` applied is required, so this is an illustrative block, not a runnable doctest):

    .. code-block:: python

        from silly_kicks.vaep.adjusted import adjusted_value

        adj = adjusted_value(actions, p_scores_success, p_concedes_fail, xsuccess)
        adj["vaep_value"].sum()   # total outcome-bias-free VAEP over the actions
    """
    xs = np.asarray(xsuccess, dtype=float)
    pss = np.asarray(p_scores_success, dtype=float)
    pcf = np.asarray(p_concedes_fail, dtype=float)
    p_scores_adj = pd.Series(xs * pss, index=actions.index)
    p_concedes_adj = pd.Series((1.0 - xs) * pcf, index=actions.index)
    return formula.value(actions, p_scores_adj, p_concedes_adj)
