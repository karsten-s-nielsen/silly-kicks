"""Shared training / completion-metric exclusion mask (TF-46 Arm 1, decision B5/E).

One helper decides whether a SPADL action row must be excluded from the GK-completion training set (G1),
the xsuccess training set, and delivery / completion metrics. It reads the base signal columns
``result_source`` and ``restart_evidence`` and the SkillCorner-specific ``repair_kind``.

The helper **tolerates** a missing column: absence contributes an all-False sub-mask, never a raise
(decision E). The anti-silent-leak guarantee is structural and lives at the producer -- ``_finalize_output``
strict projection raises if a converter omits a base column, so any converter output always carries
``result_source`` / ``restart_evidence``; a frame that lacks them (atomic SPADL, a hand-built frame, a
pre-re-convert frame) has no marked rows to leak, because a row cannot be marked in a column that does not
exist.
"""

from __future__ import annotations

import pandas as pd

#: ``result_source`` values that mark a non-native (estimated) result.
_EXCLUDED_RESULT_SOURCES = frozenset({"inferred", "stopgap"})
#: ``restart_evidence`` value that marks an ambiguous chained restart.
_EXCLUDED_RESTART_EVIDENCE = frozenset({"chain_ambiguous"})


def training_exclusion_mask(actions: pd.DataFrame) -> pd.Series:
    """Return a boolean Series, True where the row must be excluded from training and completion metrics.

    Excludes a row when any of:

    * ``result_source`` in {``inferred``, ``stopgap``};
    * ``restart_evidence == "chain_ambiguous"``;
    * ``repair_kind`` is set (a non-``<NA>`` goals-reconciliation repair row).

    ``<NA>`` is kept (not excluded). A column that is absent from ``actions`` contributes nothing to the
    mask (tolerant; see module docstring). The returned Series is index-aligned to ``actions`` and never
    raises on a missing column.
    """
    mask = pd.Series(False, index=actions.index)

    if "result_source" in actions.columns:
        mask = mask | actions["result_source"].isin(_EXCLUDED_RESULT_SOURCES)
    if "restart_evidence" in actions.columns:
        mask = mask | actions["restart_evidence"].isin(_EXCLUDED_RESTART_EVIDENCE)
    if "repair_kind" in actions.columns:
        mask = mask | actions["repair_kind"].notna()

    return mask.fillna(False).astype(bool)
