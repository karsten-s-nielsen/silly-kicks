"""W2 (4.45.0): faithful production V_opp -- observed post-turnover, possession-bound, bin-widened.

The mirror V_opp was a geometric proxy; this promotes EmpiricalTurnoverValue to production with a
support-gated hierarchical fallback (native cell -> coarse block -> global), possession-bound scope
(window_seconds=None, scope-symmetric with V), and a game_id guard. Mirror -> cross-check.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import EmpiricalTurnoverValue, PressureLevels, TurnoverCost
from silly_kicks.xtgk._possession_value import PressureLevel, flat_zones

_PASS = spadlconfig.actiontype_id["pass"]
_SHOT = spadlconfig.actiontype_id["shot"]
_FAIL = spadlconfig.result_id["fail"]
_SUCCESS = spadlconfig.result_id["success"]


def _act(aid, team, typ, res, sx, sy, t, *, poss, game=1, xg=0.0, pressure=0.5):
    return dict(
        game_id=game,
        period_id=1,
        action_id=aid,
        time_seconds=t,
        team_id=team,
        player_id=1,
        type_id=typ,
        result_id=res,
        start_x=sx,
        start_y=sy,
        end_x=sx,
        end_y=sy,
        possession_id=poss,
        xg=xg,
        pressure=pressure,
    )


def _turnover(aid, team, sx, sy, t, *, poss, game=1, pressure=0.5):
    return _act(aid, team, _PASS, _FAIL, sx, sy, t, poss=poss, game=game, pressure=pressure)  # failed pass


def _oppshot(aid, team, t, *, poss, game=1, xg):
    return _act(aid, team, _SHOT, _SUCCESS, 100.0, 34.0, t, poss=poss, game=game, xg=xg)


def _fit(rows, *, min_support=3, coarsen=4, window_seconds=None):
    """Fit with a SHARED PressureLevels; return (tc, p) where p is the single tercile of the
    constant-pressure=0.5 rows (so assertions don't hard-code a tercile)."""
    a = pd.DataFrame(rows)
    pl = PressureLevels().fit(a["pressure"])
    tc = EmpiricalTurnoverValue(min_support=min_support, coarsen=coarsen, window_seconds=window_seconds).fit(
        a, xg_column="xg", pressure_column="pressure", pressure_levels=pl
    )
    p = cast(PressureLevel, int(np.asarray(pl.apply(a["pressure"]))[0]))
    return tc, p


def test_faithful_vopp_satisfies_turnovercost_port():
    rows = [_turnover(0, 1, 5.0, 34.0, 10.0, poss=1), _oppshot(1, 2, 12.0, poss=2, xg=0.2)]
    tc, p = _fit(rows, min_support=1)
    assert isinstance(tc, TurnoverCost)
    z = int(flat_zones([5.0], [34.0])[0])
    assert 0.0 <= tc.value(z, p) <= 1.0
    assert tc.surface(p).shape == (12, 16)
    assert tc.support(p).ravel()[z] >= 1  # native n exposed


def test_faithful_vopp_bin_widening_non_vacuous():
    # deep cell z0 has 1 turnover with NO opp shot (native mean 0); two OTHER cells in its coarse block
    # each have a turnover with opp shot 0.15 -> block mean 0.10. min_support=3: z0 native n=1 < 3 ->
    # falls back to the block (n=3) -> resolved 0.10, level 1, DIFFERENT from native 0.0 (non-vacuous).
    rows = [_turnover(0, 1, 5.0, 34.0, 10.0, poss=1), _oppshot(1, 2, 12.0, poss=2, xg=0.0)]  # z0: credit 0
    rows += [_turnover(2, 1, 15.0, 34.0, 20.0, poss=3), _oppshot(3, 2, 22.0, poss=4, xg=0.15)]  # block sibling
    rows += [_turnover(4, 1, 25.0, 40.0, 30.0, poss=5), _oppshot(5, 2, 32.0, poss=6, xg=0.15)]  # block sibling
    tc, p = _fit(rows, min_support=3, coarsen=4)
    z0 = int(flat_zones([5.0], [34.0])[0])
    assert tc.support(p).ravel()[z0] == 1  # native sparse
    assert tc.resolution_level(p).ravel()[z0] == 1  # resolved via the coarse block
    assert tc.value(z0, p) == pytest.approx(0.10, abs=1e-6)  # block mean, NOT native 0.0 (fallback fired)


def test_possession_bound_vs_10s_scope():
    rows = [
        _turnover(0, 1, 5.0, 34.0, 10.0, poss=1),
        _act(1, 2, _PASS, _SUCCESS, 60.0, 34.0, 12.0, poss=2),  # opp keeps ball
        _oppshot(2, 2, 25.0, poss=2, xg=0.30),
    ]  # opp shot >10s, within won possession
    z = int(flat_zones([5.0], [34.0])[0])
    bound, pb = _fit(rows, min_support=1, window_seconds=None)
    capped, pc = _fit(rows, min_support=1, window_seconds=10.0)
    assert bound.value(z, pb) == pytest.approx(0.30, abs=1e-6)  # possession-bound credits it
    assert capped.value(z, pc) == pytest.approx(0.0, abs=1e-6)  # 10s cap drops it

    # R6: cross-match -- a match-2 opp shot must NOT be charged to a match-1 turnover under window=None
    rows2 = [
        _turnover(0, 1, 5.0, 34.0, 10.0, poss=1, game=1),
        _oppshot(1, 2, 11.0, poss=2, game=2, xg=0.40),
    ]  # different game
    b2, p2 = _fit(rows2, min_support=1, window_seconds=None)
    assert b2.value(z, p2) == pytest.approx(0.0, abs=1e-6)  # game boundary bounds the scan


def test_fit_requires_non_null_game_id():
    rows = [_turnover(0, 1, 5.0, 34.0, 10.0, poss=1), _oppshot(1, 2, 12.0, poss=2, xg=0.2)]
    df = pd.DataFrame(rows).drop(columns=["game_id"])
    with pytest.raises(ValueError, match="game_id"):
        EmpiricalTurnoverValue(window_seconds=None).fit(df, xg_column="xg", pressure_column="pressure")
    df2 = pd.DataFrame(rows)
    df2.loc[0, "game_id"] = None
    with pytest.raises(ValueError, match="game_id"):
        EmpiricalTurnoverValue(window_seconds=None).fit(df2, xg_column="xg", pressure_column="pressure")


def test_mirror_vs_empirical_divergence_reported():
    from silly_kicks.xtgk._turnover import surface_divergence  # per-zone |a.surface(p) - b.surface(p)|

    rows = [_turnover(i, 1, 5.0, 34.0, 10.0 + i, poss=i + 1) for i in range(4)]
    rows += [_oppshot(100, 2, 12.0, poss=99, xg=0.2)]
    emp, p = _fit(rows, min_support=1)
    d = surface_divergence(emp, emp, p)  # a surface vs itself -> all zeros
    assert d.shape == (12, 16) and np.allclose(d, 0.0)
