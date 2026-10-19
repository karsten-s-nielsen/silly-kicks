"""Per-routine chance quality -- descriptive corner effectiveness (TF-46 Arm 1 value-add).

Shaw explicitly DEFERRED "which routines are most effective". This is the offence analog of Arm 2's
zonal-config-effectiveness primitive: shots / goals / summed xG inside each corner's outcome window
``[t_kick, t_outcome]``, and a soft-weighted (by NMF coefficient) mean of those per routine feature.

DESCRIPTIVE only -- an outcome-association, NOT a causal routine valuation (that is Arm 3): it controls
for neither the defensive setup nor finishing skill. xG is an INJECTED port (silly-kicks ships no xG
model). A corner with no resolvable outcome keeps its row with zeros, never dropped (ADR-042).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig

from ._columns import CC_N_GOALS, CC_N_SHOTS, CC_SOURCE, CC_SUM_XG, CORNER_CHANCE_COLUMNS, CORNER_KEYS
from ._config import ChanceQualityParams

_DEFAULT_CHANCE_PARAMS = ChanceQualityParams()
_SHOT_TYPE_IDS = frozenset(spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick"))
_SUCCESS = spadlconfig.result_id["success"]


def compute_corner_chance_quality(
    corners: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    xg_column: str | None = None,
    params: ChanceQualityParams = _DEFAULT_CHANCE_PARAMS,
) -> pd.DataFrame:
    """Per-corner shots / goals / summed xG inside ``[t_kick, t_outcome]``. Grain ``(game_id, corner_id)``.

    ``n_shots`` and ``n_goals`` are MODEL-FREE (fully observed) and are always computed. ``sum_xg`` needs
    an injected xG column (silly-kicks ships no xG model, ADR-110/ADR-085): pass ``xg_column`` to populate
    it, else ``sum_xg`` is honest-NaN (never a fabricated 0.0). A NAMED-but-absent ``xg_column`` still
    raises -- that is a caller error, distinct from "this provider supplies no xG".

    Examples
    --------
    Measure the chance quality each corner produced from an injected xG column::

        from silly_kicks.corners import compute_corner_chance_quality
        quality = compute_corner_chance_quality(windowed_corners, actions, xg_column="xg")
    """
    if xg_column is not None and xg_column not in actions.columns:
        raise KeyError(
            f"xg_column {xg_column!r} is not in actions -- silly-kicks ships no xG model. Inject an xG "
            "column (port pattern; cf. xg_column in vaep/labels.py), or omit xg_column for n_shots/n_goals only."
        )
    xg = pd.to_numeric(actions[xg_column], errors="coerce") if xg_column is not None else None
    is_shot = actions["type_id"].isin(_SHOT_TYPE_IDS)
    is_goal = is_shot & (actions["result_id"] == _SUCCESS)

    rows: list[dict] = []
    for _, c in corners.iterrows():
        period = int(c["period_id"])
        t0 = float(c["t_kick"])
        t1 = float(c["t_outcome"]) if "t_outcome" in corners.columns and pd.notna(c["t_outcome"]) else t0 + 6.0
        in_win = (actions["period_id"] == period) & (actions["time_seconds"] >= t0) & (actions["time_seconds"] <= t1)
        shot_win = in_win & is_shot
        if xg is None:
            sum_xg = float("nan")  # no xG injected -> honest-NaN, never a fabricated 0.0
        else:
            sum_xg = float(np.nansum(xg[shot_win].to_numpy(dtype=float))) if shot_win.any() else 0.0
        rows.append(
            {
                "game_id": c["game_id"],
                "corner_id": int(c["corner_id"]),
                CC_N_SHOTS: int(shot_win.sum()),
                CC_N_GOALS: int((in_win & is_goal).sum()),
                CC_SUM_XG: sum_xg,
                CC_SOURCE: xg_column,
            }
        )
    cols = list(CORNER_CHANCE_COLUMNS)
    if not rows:
        return pd.DataFrame({cc: pd.Series(dtype=t) for cc, t in CORNER_CHANCE_COLUMNS.items()})
    return pd.DataFrame(rows, columns=cols).astype(CORNER_CHANCE_COLUMNS)


def aggregate_chance_quality_by_feature(features: pd.DataFrame, chance_quality: pd.DataFrame) -> pd.DataFrame:
    """Soft-weighted (by routine-feature coefficient) mean chance quality per routine feature.

    Descriptive aggregation -- NOT a ranking (which-routine-beats-which stays consumer-side, ADR-110).
    One row per ``routine_feat_<k>`` with the coefficient-weighted mean of each chance metric.

    Examples
    --------
    Average chance quality per routine feature, coefficient-weighted::

        by_feature = aggregate_chance_quality_by_feature(routine_features, chance_quality)
    """
    feat_cols = [c for c in features.columns if c.startswith("routine_feat_")]
    joined = features.merge(chance_quality, on=CORNER_KEYS, how="inner")
    metrics = [CC_N_SHOTS, CC_N_GOALS, CC_SUM_XG]
    out_rows: list[dict] = []
    for fc in feat_cols:
        w = joined[fc].to_numpy(dtype=float)
        valid = ~np.isnan(w)
        wsum = float(w[valid].sum())
        row: dict = {"routine_feature": fc, "weight_sum": wsum}
        for m in metrics:
            vals = joined[m].to_numpy(dtype=float)
            row[f"weighted_mean_{m}"] = float((w[valid] * vals[valid]).sum() / wsum) if wsum > 0 else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows, columns=["routine_feature", "weight_sum", *[f"weighted_mean_{m}" for m in metrics]])
