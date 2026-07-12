"""T5 -- actions and retention_features MUST describe the same coordinates (ADR-036 amendment).

Closes F1 (resolved actions + raw features) AND its mirror R1 (raw actions + resolved features).
The check compares COORDINATES, not provenance, so both directions -- and mart-vintage divergence --
fall out of one rule with no case table.
"""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import PressureLevels, apply_resolved_gk_geometry, compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV
from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN
from silly_kicks.xtgk._retention_features import extract_retention_features

GOALKICK = spadlconfig.actiontype_id["goalkick"]


class _StubV:
    def value(self, zone, p):
        return 0.02

    def surface(self, p):
        return np.full((12, 16), 0.02)

    def delta_v(self, s, s_next):
        return DeltaV(delta=0.03, pressure_component=0.0, position_component=0.03)


class _StubRho:
    def predict_proba(self, features):
        return np.full(len(features), 0.8)


class _StubTurnover:
    def value(self, zone, p):
        return 0.05

    def surface(self, p):
        return np.full((12, 16), 0.05)

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _raw():
    """A SkillCorner-shaped row: raw origin PRESENT and WRONG; identical end_x in both frames (the
    dest override is a measured no-op on real data) -- so ONLY the origin diverges."""
    return pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [0],
            "type_id": [GOALKICK],
            "is_gk_distribution": [True],
            "start_x": [25.0],
            "start_y": [40.0],
            "end_x": [40.0],
            "end_y": [34.0],
            "xt_gk_origin_x": [4.29],
            "xt_gk_origin_y": [34.0],
            "xt_gk_dest_x": [40.0],
            "xt_gk_dest_y": [34.0],
            "pressure": [0.1],
        }
    )


def _call(actions, feats):
    return compute_xt_gk_v2(
        actions,
        possession_value=_StubV(),
        retention=_StubRho(),
        turnover_cost=_StubTurnover(),
        pressure_levels=PressureLevels().fit(pd.Series([0.0, 0.1, 1.0])),
        retention_features=feats,
    )


def test_f1_resolved_actions_with_raw_features_raises():
    raw = _raw()
    resolved = apply_resolved_gk_geometry(raw)
    with pytest.raises(ValueError, match="coordinates"):
        _call(resolved, extract_retention_features(raw))


def test_r1_mirror_raw_actions_with_resolved_features_raises():
    raw = _raw()
    resolved = apply_resolved_gk_geometry(raw)
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="coordinates"):
        _call(raw, extract_retention_features(resolved))


def test_origin_only_divergence_is_caught_end_x_is_identical():
    """Proves the check spans length/forwardness/dy_abs and is NOT dest_x-only. The dest override
    is a measured no-op on real data, so a dest_x-only check would miss every real divergence."""
    raw = _raw()
    resolved = apply_resolved_gk_geometry(raw)
    assert float(raw.iloc[0]["end_x"]) == float(resolved.iloc[0]["end_x"])  # dest identical
    assert float(raw.iloc[0]["start_x"]) != float(resolved.iloc[0]["start_x"])  # origin differs
    with pytest.raises(ValueError, match="coordinates"):
        _call(resolved, extract_retention_features(raw))


def test_coherent_pair_scores_normally():
    resolved = apply_resolved_gk_geometry(_raw())
    out = _call(resolved, extract_retention_features(resolved))
    assert np.isfinite(out.iloc[0]["xt_gk_v2"])


def test_unstamped_actions_with_a_gk_domain_warn_once_and_still_score():
    raw = _raw()  # never passed through the helper -> no stamp column at all
    with pytest.warns(UserWarning, match="apply_resolved_gk_geometry"):
        out = _call(raw, extract_retention_features(raw))
    assert np.isfinite(out.iloc[0]["xt_gk_v2"])


def test_unattested_stamped_actions_also_warn():
    """S1: the R2 semantics ("`unattested` is treated as unstamped for warning purposes") shipped
    UNTESTED. A frame that WENT THROUGH the helper but found no resolved columns is stamped
    `unattested`, and must still warn -- otherwise it scores raw origins in silence."""
    raw = _raw().drop(columns=["xt_gk_origin_x", "xt_gk_origin_y", "xt_gk_dest_x", "xt_gk_dest_y"])
    with pytest.warns(UserWarning):  # the helper itself warns about the missing columns
        stamped = apply_resolved_gk_geometry(raw)
    assert stamped.iloc[0][GK_GEOMETRY_SOURCE_COLUMN] == "unattested"

    with pytest.warns(UserWarning, match="apply_resolved_gk_geometry"):
        out = _call(stamped, extract_retention_features(stamped))
    assert np.isfinite(out.iloc[0]["xt_gk_v2"])


def test_mixed_vintage_frame_warns_when_only_some_rows_are_unattested():
    """S1 again, the case `.all()` would MISS: a concatenated frame where one row is attested and
    one is not. `.any()` must fire, else the unattested row scores raw origins silently."""
    resolved = apply_resolved_gk_geometry(_raw())
    unattested = resolved.copy()
    unattested[GK_GEOMETRY_SOURCE_COLUMN] = "unattested"
    unattested["action_id"] = [1]
    mixed = pd.concat([resolved, unattested], ignore_index=True)

    with pytest.warns(UserWarning, match="apply_resolved_gk_geometry"):
        _call(mixed, extract_retention_features(mixed))
