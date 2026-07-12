"""T1 -- compute_xt_gk_v2 NEVER fabricates a zone from a NaN coordinate (ADR-036 amendment).

The defect (4.40.0-4.45.0): flat_zones maps NaN -> (0,0) -> zone 176 (the own-corner cell), and the
scoring seam dropped nothing, so a NaN-origin goal-kick was scored as a REAL number at a fabricated
location. ~24% of the Gradient Sports GK-distribution domain.
"""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import PressureLevels, compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV

GOALKICK = spadlconfig.actiontype_id["goalkick"]
_OUT = ["xt_gk_v2_position", "xt_gk_v2_pev", "xt_gk_v2_retention_loss", "xt_gk_v2_dzv", "xt_gk_v2"]


class _StubV:
    def value(self, zone, p):
        return 0.02

    def surface(self, p):
        return np.full((12, 16), 0.02)

    def delta_v(self, s, s_next):
        return DeltaV(delta=0.03, pressure_component=0.0, position_component=0.03)


class _CountingRho:
    """Records how many rows it was asked to score -- the NaN rows must never reach it."""

    def __init__(self):
        self.seen = 0

    def predict_proba(self, features):
        self.seen += len(features)
        return np.full(len(features), 0.8)


class _StubTurnover:
    def value(self, zone, p):
        return 0.05

    def surface(self, p):
        return np.full((12, 16), 0.05)

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _actions(start_x):
    n = len(start_x)
    return pd.DataFrame(
        {
            "game_id": [1] * n,
            "period_id": [1] * n,
            "action_id": list(range(n)),
            "type_id": [GOALKICK] * n,
            "start_x": start_x,
            "start_y": [34.0] * n,
            "end_x": [40.0] * n,
            "end_y": [34.0] * n,
            "pressure": [0.1] * n,
        }
    )


def _call(actions, rho):
    from silly_kicks.xtgk._retention_features import extract_retention_features

    return compute_xt_gk_v2(
        actions,
        possession_value=_StubV(),
        retention=rho,
        turnover_cost=_StubTurnover(),
        pressure_levels=PressureLevels().fit(pd.Series([0.0, 0.1, 1.0])),
        retention_features=extract_retention_features(actions),
    )


def test_nan_coord_row_emits_nan_not_zone_176_value():
    """The defect: NaN -> flat_zones -> zone 176 -> a REAL number. Now it must be NaN."""
    actions = _actions([5.5, np.nan])
    with pytest.warns(UserWarning):
        out = _call(actions, _CountingRho())
    assert np.isfinite(out.iloc[0]["xt_gk_v2"])  # the finite row still scores
    for c in _OUT:
        assert np.isnan(out.iloc[1][c]), f"{c} was fabricated for a NaN-coord row"


def test_finite_rows_are_byte_identical_to_the_all_finite_run():
    """The guard must not perturb a single finite row. The claim is BYTE-identity, so this asserts
    exact `==`, not pytest.approx -- the code path supports it."""
    clean = _actions([5.5, 6.0])
    mixed = _actions([5.5, np.nan])
    out_clean = _call(clean, _CountingRho())
    with pytest.warns(UserWarning):
        out_mixed = _call(mixed, _CountingRho())
    for c in _OUT:
        assert out_mixed.iloc[0][c] == out_clean.iloc[0][c]


def test_rho_is_never_called_on_non_finite_rows():
    """Closes the silent mean-imputation path (_retention.py:81) without touching predict_proba."""
    rho = _CountingRho()
    with pytest.warns(UserWarning):
        _call(_actions([5.5, np.nan, np.nan]), rho)
    assert rho.seen == 1, f"rho scored {rho.seen} rows; only the 1 finite row should reach it"


def test_warns_with_a_count_of_dropped_rows():
    with pytest.warns(UserWarning, match="2 of 3"):
        _call(_actions([5.5, np.nan, np.nan]), _CountingRho())
