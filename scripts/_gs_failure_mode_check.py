"""Task 9 (R6): GS failure-mode tagging reliability -- a pre-run check before the split is trusted.

``classify_failure_mode`` routes an intercepted failure to the receiver-model target and an out-of-play
failure to the trajectory target. If GS's next-action tagging cannot cleanly separate the two (too many
failed passes classify as ``other``), the routing is unreliable and the driver must NOT mix the legs --
so this check gates it, on the pinned ``MAX_AMBIGUOUS_RATE``.
"""

from __future__ import annotations

import pandas as pd

from scripts._cover_shadow_thresholds import MAX_AMBIGUOUS_RATE
from scripts._receiver_validation import _PASS_TYPES, _R, classify_failure_mode


def failure_mode_reliability(actions: pd.DataFrame, *, max_ambiguous_rate: float = MAX_AMBIGUOUS_RATE) -> dict:
    """Fraction of failed passes whose failure mode is ``other`` (un-classifiable); ``reliable`` iff below the bar."""
    fmode = classify_failure_mode(actions)
    is_fp = actions["type_id"].isin(_PASS_TYPES) & (actions["result_id"] == _R["fail"])
    modes = fmode.reindex(actions.loc[is_fp, "action_id"].to_numpy())
    n = len(modes)
    n_ambiguous = int((modes == "other").sum())
    rate = n_ambiguous / n if n else float("nan")
    return {
        "n_failed": n,
        "n_ambiguous": n_ambiguous,
        "ambiguous_rate": rate,
        "reliable": bool(n and rate < max_ambiguous_rate),
    }
