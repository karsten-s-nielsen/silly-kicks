"""Shared training / completion-metric exclusion mask (TF-46 Arm 1, decision B5/E).

One helper decides whether a SPADL action row must be excluded from the GK-completion training set (G1),
the xsuccess training set, and delivery / completion metrics. It reads the base signal columns
``result_source``, ``restart_evidence`` and ``repair_kind`` (``repair_kind`` was promoted to base
``SPADL_COLUMNS`` in TF-46 Arm 1 Part 2, decision D1).

The helper **tolerates** a missing column: absence contributes an all-False sub-mask, never a raise
(decision E). The anti-silent-leak guarantee is structural and lives at the producer -- ``_finalize_output``
strict projection raises if a converter omits a base column, so any converter output always carries
``result_source`` / ``restart_evidence``; a frame that lacks them (atomic SPADL, a hand-built frame, a
pre-re-convert frame) has no marked rows to leak, because a row cannot be marked in a column that does not
exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: ``result_source`` values that mark a non-native (estimated) result.
_EXCLUDED_RESULT_SOURCES = frozenset({"inferred", "stopgap"})
#: ``restart_evidence`` value that marks an ambiguous chained restart.
_EXCLUDED_RESTART_EVIDENCE = frozenset({"chain_ambiguous"})

#: The provenance dimensions (TF-46 Arm 1 Part 2, spec section 18.2.1). A training entry point declares which
#: it READS (``TRAINING_READS``); the declaration gate checks that against its column literals.
DIMENSIONS = ("result", "time", "type", "location", "existence")


def _col(actions: pd.DataFrame, name: str) -> pd.Series:
    return actions[name] if name in actions.columns else pd.Series([pd.NA] * len(actions), index=actions.index)


def _gated_t_kick(actions: pd.DataFrame) -> pd.Series:
    """True where resolve_kicks SHIPPED a kick time (``kick_time_source`` not withheld/none, ``t_kick`` finite)."""
    if "t_kick" not in actions.columns or "kick_time_source" not in actions.columns:
        return pd.Series(False, index=actions.index)
    kts = _col(actions, "kick_time_source")
    shipped = ~kts.isin(["withheld", "none"]) & kts.notna()
    return (shipped & actions["t_kick"].notna()).fillna(False).astype(bool)


def _gated_penalty(actions: pd.DataFrame) -> pd.Series:
    """A valid-time penalty row: a Sportec native penalty, or a Part-2 gated-detection penalty row."""
    try:
        from silly_kicks.spadl import config as _cfg

        sp = _cfg.actiontype_id.get("shot_penalty")
    except Exception:
        sp = None
    if sp is None or "type_id" not in actions.columns:
        return pd.Series(False, index=actions.index)
    is_pen = actions["type_id"] == sp
    native = _col(actions, "type_source") == "native"
    gated = _col(actions, "penalty_evidence").notna()
    return (is_pen & (native | gated)).fillna(False).astype(bool)


def _result_unvalidated(actions: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=actions.index)
    if "result_source" in actions.columns:
        mask = mask | actions["result_source"].isin(_EXCLUDED_RESULT_SOURCES)
    if "restart_evidence" in actions.columns:
        mask = mask | actions["restart_evidence"].isin(_EXCLUDED_RESTART_EVIDENCE)
    if "repair_kind" in actions.columns:
        mask = mask | actions["repair_kind"].notna()
    return mask.fillna(False).astype(bool)


def provenance_masks(actions: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-dimension unvalidated masks (spec section 18.2.1). Each value is a boolean Series, True where the
    row is UNVALIDATED in that dimension. Tolerant of missing columns (an absent column contributes False)."""
    n_idx = actions.index
    derived = (_col(actions, "action_provenance") == "derived").fillna(False).astype(bool)
    repair = _col(actions, "repair_kind").notna().fillna(False).astype(bool)
    gated_tk = _gated_t_kick(actions)
    pen_valid = _gated_penalty(actions)

    result = _result_unvalidated(actions)
    # time: a derived restart without a gated t_kick, or a repair row (episode-bounded); penalties are valid.
    time = (((derived & ~gated_tk) | repair) & ~pen_valid).fillna(False).astype(bool)
    # type: default-typed rows (the subtype is a guess).
    type_ = (_col(actions, "type_source") == "default").fillna(False).astype(bool)
    # location: canonical start NaN and no gated enriched origin.
    start_nan = ~np.isfinite(pd.to_numeric(_col(actions, "start_x"), errors="coerce").to_numpy(dtype="float64"))
    origin_gated = _col(actions, "origin_evidence").notna().fillna(False).astype(bool)
    location = pd.Series(start_nan, index=n_idx) & ~origin_gated
    existence = pd.Series(False, index=n_idx)  # none today (every derived row records a real event)
    return {"result": result, "time": time, "type": type_, "location": location.astype(bool), "existence": existence}


def uncertainty_interval(actions: pd.DataFrame) -> pd.DataFrame:
    """``t_lo`` / ``t_hi`` per row: the event's time-uncertainty interval, NaN where the time is validated
    (spec section 18.2.1). A derived restart's interval is its kick bracket; a repair row's is its goal
    episode (approximated here by the kick bracket when present, else the row time)."""
    time_unval = provenance_masks(actions)["time"].to_numpy(dtype=bool)
    t = pd.to_numeric(_col(actions, "time_seconds"), errors="coerce").to_numpy(dtype="float64")
    lo = pd.to_numeric(_col(actions, "kick_bracket_lo"), errors="coerce").to_numpy(dtype="float64")
    hi = pd.to_numeric(_col(actions, "kick_bracket_hi"), errors="coerce").to_numpy(dtype="float64")
    lo = np.where(np.isfinite(lo), lo, t)
    hi = np.where(np.isfinite(hi), hi, t)
    return pd.DataFrame(
        {"t_lo": np.where(time_unval, lo, np.nan), "t_hi": np.where(time_unval, hi, np.nan)}, index=actions.index
    )


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
