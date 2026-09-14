"""option_value -- pluggable typed family. Default: completion x progression (Eyestone-validated).

EV = completion * (1 + max(0, opponents_bypassed)). A backward/lateral option (bypassed <= 0) ->
EV = completion. NaN in either input -> NaN EV (never fabricated). xT/retention variants are reserved
typed doors (NotImplementedError) -- not implemented in PR1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._config import GkDecisionParams


def option_value(rows: pd.DataFrame, *, params: GkDecisionParams) -> pd.Series:
    """EV per option row from ``completion`` and ``opponents_bypassed`` (float64; NaN-propagating).

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.gk_decision import GkDecisionParams, option_value
    >>> rows = pd.DataFrame({"completion": [0.9, 0.9], "opponents_bypassed": [0.0, 3.0]})
    >>> option_value(rows, params=GkDecisionParams()).tolist()
    [0.9, 3.6]
    """
    if params.value_fn != "completion_progression":
        raise NotImplementedError(
            f"value_fn={params.value_fn!r} is a reserved door (PR1 ships only 'completion_progression')"
        )
    comp = pd.to_numeric(rows["completion"], errors="coerce").to_numpy(dtype="float64")
    byp = pd.to_numeric(rows["opponents_bypassed"], errors="coerce").to_numpy(dtype="float64")
    ev = comp * (1.0 + np.clip(byp, 0.0, None))  # NaN propagates through * and clip
    return pd.Series(ev, index=rows.index, dtype="float64")
