"""xImpact (TF-63): VAEP_adjusted weighted by in-game goal leverage.

``xImpact(action) = VAEP_adjusted(action) x dP(win | goal at the pre-action state)``.

``VAEP.rate_ximpact`` (in ``vaep/base.py``) computes ``VAEP_adjusted`` via the existing
``rate_adjusted`` (which raises on HybridVAEP), reads the leverage from ``win_probability.goal_leverage``
(a function-local import -- the layering edge), aligns on the action index, and calls this helper. See
NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import pandas as pd


def ximpact_values(adjusted: pd.DataFrame, leverage: pd.Series) -> pd.Series:
    """``VAEP_adjusted`` (the ``vaep_value`` column) x ``dP(win | goal)``, index-aligned, NaN-propagating.

    Parameters
    ----------
    adjusted : pandas.DataFrame
        Output of :meth:`silly_kicks.vaep.VAEP.rate_adjusted` (carries ``vaep_value``).
    leverage : pandas.Series
        ``ΔP(win | goal)`` per action from :func:`silly_kicks.win_probability.goal_leverage`, on the
        same action index. A NaN leverage (unresolved state) propagates to a NaN xImpact.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.vaep.ximpact import ximpact_values
    >>> adjusted = pd.DataFrame({"vaep_value": [0.10, -0.04]})
    >>> leverage = pd.Series([0.5, 0.2])
    >>> ximpact_values(adjusted, leverage).round(4).tolist()
    [0.05, -0.008]
    """
    v = adjusted["vaep_value"]
    lev = leverage.reindex(v.index)
    return pd.Series(v.to_numpy() * lev.to_numpy(), index=v.index, name="ximpact")
