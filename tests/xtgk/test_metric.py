import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import PressureLevels
from silly_kicks.xtgk._metric import compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV

GOALKICK = spadlconfig.actiontype_id["goalkick"]


class _StubV:
    def __init__(self, surface_value=0.02):
        self._val = surface_value

    def value(self, zone, p):
        return self._val

    def surface(self, p):
        return np.full((12, 16), self._val)

    def delta_v(self, s, s_next):
        # position-only ΔV (p'=p) of +0.03; pressure component 0
        return DeltaV(delta=0.03, pressure_component=0.0, position_component=0.03)


class _StubRho:
    def __init__(self, rho):
        self._rho = rho

    def predict_proba(self, features):
        return np.full(len(features), self._rho)


class _StubTurnover:
    def __init__(self, v_opp=0.05):
        self._v = v_opp

    def value(self, zone, p):
        return self._v

    def surface(self, p):
        return np.full((12, 16), self._v)

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _one_goalkick():
    return pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                time_seconds=0.0,
                team_id=10,
                player_id=1,
                type_id=GOALKICK,
                result_id=spadlconfig.result_id["success"],
                bodypart_id=0,
                start_x=5.0,
                start_y=34.0,
                end_x=40.0,
                end_y=34.0,
                pressure=0.1,
            )
        ]
    )


def _pl_and_feats(actions):
    pl = PressureLevels().fit(actions["pressure"])
    feats = pd.DataFrame(index=actions.index)  # stub rho ignores content, uses len(features)
    return pl, feats


def test_four_terms_sum_to_metric_and_pev_zero_when_pprime_equals_p():
    actions = _one_goalkick()
    pl, feats = _pl_and_feats(actions)
    out = compute_xt_gk_v2(
        actions,
        possession_value=_StubV(),
        retention=_StubRho(0.8),
        turnover_cost=_StubTurnover(),
        kappa=1.0,
        pressure_column="pressure",
        pressure_levels=pl,
        retention_features=feats,
    )
    row = out.iloc[0]
    # terms: (1) 0.8*0.03  (2) 0.8*0.0=PEV  (3) -0.2*0.02  (4) -0.2*1.0*0.05
    assert np.isclose(row["xt_gk_v2_position"], 0.8 * 0.03)
    assert np.isclose(row["xt_gk_v2_pev"], 0.0)
    assert np.isclose(row["xt_gk_v2_retention_loss"], -0.2 * 0.02)
    assert np.isclose(row["xt_gk_v2_dzv"], -0.2 * 1.0 * 0.05)
    total = row["xt_gk_v2_position"] + row["xt_gk_v2_pev"] + row["xt_gk_v2_retention_loss"] + row["xt_gk_v2_dzv"]
    assert np.isclose(row["xt_gk_v2"], total)


def test_kappa_scales_only_the_dzv_term():
    actions = _one_goalkick()
    pl, feats = _pl_and_feats(actions)
    out1 = compute_xt_gk_v2(
        actions,
        possession_value=_StubV(),
        retention=_StubRho(0.8),
        turnover_cost=_StubTurnover(),
        kappa=1.0,
        pressure_column="pressure",
        pressure_levels=pl,
        retention_features=feats,
    )
    out2 = compute_xt_gk_v2(
        actions,
        possession_value=_StubV(),
        retention=_StubRho(0.8),
        turnover_cost=_StubTurnover(),
        kappa=2.0,
        pressure_column="pressure",
        pressure_levels=pl,
        retention_features=feats,
    )
    assert np.isclose(out2.iloc[0]["xt_gk_v2_dzv"], 2 * out1.iloc[0]["xt_gk_v2_dzv"])
    assert np.isclose(out2.iloc[0]["xt_gk_v2_position"], out1.iloc[0]["xt_gk_v2_position"])


def test_requires_pressure_levels_and_features():
    actions = _one_goalkick()
    _pl, feats = _pl_and_feats(actions)
    with pytest.raises(ValueError, match="pressure_levels"):
        compute_xt_gk_v2(
            actions,
            possession_value=_StubV(),
            retention=_StubRho(0.8),
            turnover_cost=_StubTurnover(),
            retention_features=feats,
        )  # no pl, stub has none
