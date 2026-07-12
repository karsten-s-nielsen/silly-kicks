"""T6 -- the fix must actually MOVE the number on the case it exists for.

A SkillCorner-shaped goal-kick (raw origin = broadcast ball at x=25, resolved = keeper at x=4.29)
must score DIFFERENTLY once resolved. A fix that is inert here would be worthless -- house lesson:
an A/B must exercise the path that can change the value.
"""

import warnings

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import PressureLevels, apply_resolved_gk_geometry, compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV, zone_of
from silly_kicks.xtgk._retention_features import extract_retention_features

GOALKICK = spadlconfig.actiontype_id["goalkick"]


class _ZoneSensitiveV:
    """V depends on the ORIGIN zone -- so a moved origin must move the score."""

    def value(self, zone, p):
        return 0.001 * (zone % 17)

    def surface(self, p):
        return np.zeros((12, 16))

    def delta_v(self, s, s_next):
        return DeltaV(
            delta=0.0,
            pressure_component=0.0,
            position_component=0.001 * (s_next.zone % 17) - 0.001 * (s.zone % 17),
        )


class _StubRho:
    def predict_proba(self, features):
        return np.full(len(features), 0.8)


class _StubTurnover:
    def value(self, zone, p):
        return 0.002 * (zone % 13)

    def surface(self, p):
        return np.zeros((12, 16))

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _skillcorner_goalkick():
    return pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [0],
            "type_id": [GOALKICK],
            "is_gk_distribution": [True],
            "start_x": [25.0],  # broadcast BALL detection -- present, finite, WRONG
            "start_y": [40.0],
            "end_x": [55.0],
            "end_y": [34.0],
            "xt_gk_origin_x": [4.29],  # the actual keeper
            "xt_gk_origin_y": [34.0],
            "xt_gk_dest_x": [55.0],
            "xt_gk_dest_y": [34.0],
            "pressure": [0.1],
        }
    )


def _score(actions):
    return compute_xt_gk_v2(
        actions,
        possession_value=_ZoneSensitiveV(),
        retention=_StubRho(),
        turnover_cost=_StubTurnover(),
        pressure_levels=PressureLevels().fit(pd.Series([0.0, 0.1, 1.0])),
        retention_features=extract_retention_features(actions),
    )


def test_resolving_a_skillcorner_goalkick_changes_its_score():
    raw = _skillcorner_goalkick()
    resolved = apply_resolved_gk_geometry(raw)

    # The origin genuinely moves to a different grid cell -- otherwise this test proves nothing.
    assert zone_of(25.0, 40.0) != zone_of(4.29, 34.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the raw leg intentionally warns (unattested)
        raw_score = float(_score(raw).iloc[0]["xt_gk_v2"])
    resolved_score = float(_score(resolved).iloc[0]["xt_gk_v2"])

    assert np.isfinite(raw_score) and np.isfinite(resolved_score)
    assert raw_score != resolved_score, (
        "resolving the origin did not change the score -- the fix is inert on the exact case "
        "(SkillCorner present-but-wrong broadcast-ball origin) it exists for"
    )
