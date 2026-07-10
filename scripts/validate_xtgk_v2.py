"""Owner-run validation suite for xT-GK v2 (ADR-036 §Part 5).

Construct validity is OUT-OF-SAMPLE (possession-parity split) and reported as LIFT over baselines --
V is (by construction) the expected first-shot xG, so absolute AUC vs a possession->shot target is
partly circular; the informative quantity is v2's margin over raw completion / destination-only V /
the v1 composite. The synthetic CI smoke uses a constant-rho stub (frames-free); the owner-run passes
the REAL calibrated rho with frames-derived retention_features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.spadl.utils import add_possessions
from silly_kicks.xtgk import MarkovPossessionValue, MirroredTurnoverCost, PressureLevels, compute_xt_gk_v2
from silly_kicks.xthreat._grid import M, N, _get_flat_indexes

_SHOT = spadlconfig.actiontype_id["shot"]


class _ConstRho:
    """Frames-free stub for the CI smoke; the owner-run injects the real GkRetentionModel."""

    def predict_proba(self, features):
        return np.full(len(features), 0.75)


def _possession_reaches_shot(actions: pd.DataFrame) -> np.ndarray:
    a = actions
    out = np.zeros(len(a), dtype=int)
    typ = a["type_id"].to_numpy()
    poss = a["possession_id"].to_numpy()
    for i in range(len(a)):
        for j in range(i, len(a)):
            if poss[j] != poss[i]:
                break
            if typ[j] == _SHOT:
                out[i] = 1
                break
    return out


def _auc(y, s) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def _destination_only_v(
    test: pd.DataFrame, v: MarkovPossessionValue, pl: PressureLevels, pressure_column: str
) -> np.ndarray:
    zd = _get_flat_indexes(test["end_x"], test["end_y"], N, M).to_numpy()
    zones_arg = (
        _get_flat_indexes(test["start_x"], test["start_y"], N, M).to_numpy() if pl.mode == "zone_conditional" else None
    )
    lv = pl.apply(test[pressure_column], zones=zones_arg)
    return np.array([v.value(int(z), int(p)) for z, p in zip(zd, lv, strict=True)])  # type: ignore[arg-type]


def _v1_composite(test: pd.DataFrame, frames: pd.DataFrame | None, *, xt=None, home_team_id=None) -> np.ndarray:
    # v1 xt-gk needs frames + a pre-fitted ExpectedThreat + home_team_id (owner-run supplies them);
    # the frames-free CI smoke cannot score it -> NaN (dropped from the AUC).
    if frames is None or xt is None or home_team_id is None:
        return np.full(len(test), np.nan)
    from silly_kicks.tracking import add_xt_gk

    return add_xt_gk(test, frames, xt, home_team_id=home_team_id)["xt_gk"].to_numpy()


def construct_validity_scores(
    actions: pd.DataFrame, *, xg_column: str, pressure_column: str, frames: pd.DataFrame | None = None
) -> dict:
    a = actions.reset_index(drop=True)
    if "possession_id" not in a.columns:
        a = add_possessions(a)
    train_mask = (a["possession_id"] % 2 == 0).to_numpy()  # out-of-sample by possession parity
    train, test = a[train_mask].copy(), a[~train_mask].copy()
    pl = PressureLevels().fit(train[pressure_column])
    v = MarkovPossessionValue().fit(train, xg_column=xg_column, pressure_column=pressure_column, pressure_levels=pl)
    tc = MirroredTurnoverCost(v)
    feats = pd.DataFrame(index=test.index)  # _ConstRho ignores content; owner-run supplies real features
    v2 = compute_xt_gk_v2(
        test,
        possession_value=v,
        retention=_ConstRho(),
        turnover_cost=tc,
        pressure_column=pressure_column,
        pressure_levels=pl,
        retention_features=feats,
    )
    y = _possession_reaches_shot(test)
    raw_completion = (test["result_id"] == spadlconfig.result_id["success"]).astype(int).to_numpy()
    return {
        "xt_gk_v2": {"auc": _auc(y, v2["xt_gk_v2"].to_numpy())},
        "raw_completion": {"auc": _auc(y, raw_completion)},
        "destination_xt": {"auc": _auc(y, _destination_only_v(test, v, pl, pressure_column))},
        "v1_composite": {"auc": _auc(y, _v1_composite(test, frames))},
        "_note": (
            "V == expected first-shot xG; target == possession-reaches-shot -> partial circularity. "
            "Read LIFT over baselines, not absolute AUC. Out-of-sample by possession-parity split. "
            "WC2018/Neuer repro: TODO (needs Jeff's old data)."
        ),
    }
