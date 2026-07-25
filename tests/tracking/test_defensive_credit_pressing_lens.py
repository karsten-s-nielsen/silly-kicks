"""Item 1 reverse-xT "position won" pressing lens (TF-51 v2, spec section 3).

T1 pins the reflection against a HAND-BUILT, doubly-asymmetric xT grid with LITERAL expectations
(the numbers put INTO the grid), so a wrong reflection or an axis/row-inversion regression is caught
-- NOT by re-invoking the production lookup on both sides. Default (pressing_lens=False) stays
byte-identical; the exhaustive-when-on guard pins the xt_pressing token.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.defensive_credit import DefensiveCreditParams, compute_defensive_credits
from silly_kicks.tracking.defensive_credit._sizing import sized_xt
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action

_PPF = frozenset({"pressure_pass_fail"})


def _hand_built_asymmetric_xt():
    """An ExpectedThreat whose grid is asymmetric in BOTH x and y, with two known cells.

    values_at_points reads ``grid[(w-1)-yj, xi]`` with ``xi=int(x/105*16)``, ``yj=int(y/68*12)``:
      (20, 20) -> xi=3,  yj=3 -> grid[8, 3]
      (85, 48) -> xi=12, yj=8 -> grid[3, 12]   (= the 180deg reflection of (20, 20))
    """
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    grid = np.zeros((12, 16), dtype=float)
    grid[8, 3] = 0.013  # the cell covering action-LTR (20, 20)
    grid[3, 12] = 0.207  # the cell covering (85, 48)
    xt.xT = grid
    return xt


def test_pressing_lens_reflects_to_the_mirror_point():
    xt = _hand_built_asymmetric_xt()
    # LITERAL expectations = the cell values set above, independent of the lookup implementation.
    assert sized_xt(20.0, 20.0, xt, pressing_lens=False) == pytest.approx(0.013)
    assert sized_xt(20.0, 20.0, xt, pressing_lens=True) == pytest.approx(0.207)  # = the (85,48) cell
    # semantic: a deep regain reflects near the opponent goal -> strictly higher xT
    assert sized_xt(20.0, 20.0, xt, pressing_lens=True) > sized_xt(20.0, 20.0, xt, pressing_lens=False)


def _turnover_scene_at(sx):
    a = one_action(
        type_name="pass",
        result_name="fail",
        start_x=sx,
        start_y=34.0,
        end_x=sx + 10.0,
        end_y=34.0,
        team_id=10,
        player_id=5,
    )
    a["shot_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA], dtype="boolean")
    a["xg"] = [np.nan]
    f = frame_with_defender(defender_x=sx + 1.0, defender_y=34.0)
    return a, f


def test_pressing_lens_off_turnover_sizing_is_xt(fitted_xt):
    a, f = _turnover_scene_at(95.0)
    out = compute_defensive_credits(a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(rules=_PPF))
    assert not out.empty
    assert set(out["sizing"]) == {"xt"}  # default byte-identical: no xt_pressing token


def test_pressing_lens_on_turnover_sizing_is_xt_pressing(fitted_xt):
    a, f = _turnover_scene_at(95.0)
    out = compute_defensive_credits(
        a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(pressing_lens=True, rules=_PPF)
    )
    assert not out.empty
    assert set(out["sizing"]) == {"xt_pressing"}  # EXHAUSTIVE (equals, not subset)


def test_pressing_lens_does_not_touch_shot_sizing(fitted_xt):
    # the lens only affects the xT-sized turnover rules; shot rows stay xg even with the lens on.
    a = one_action(type_name="shot", result_name="fail", start_x=95.0, start_y=34.0, team_id=10, player_id=5)
    a["shot_blocked"] = pd.array([False], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([False], dtype="boolean")
    a["xg"] = [0.2]
    f = frame_with_defender(defender_x=96.0, defender_y=34.0)
    out = compute_defensive_credits(
        a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(pressing_lens=True)
    )
    assert set(out["sizing"]) == {"xg"}


def test_pressing_lens_revalues_a_deep_regain_higher(fitted_xt):
    # fitted_xt is xT ∝ x. A deep-own-half regain (x=10) has near-zero xT(origin); its reflection
    # (x=95) sits near the opponent goal -> the lens sizes it MUCH higher (a semantic + non-vacuity check).
    a, f = _turnover_scene_at(10.0)
    off = compute_defensive_credits(a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(rules=_PPF))
    on = compute_defensive_credits(
        a, f, xg_column="xg", xt=fitted_xt, params=DefensiveCreditParams(pressing_lens=True, rules=_PPF)
    )
    off_plus = off[off["signed_value"] > 0]["signed_value"].iloc[0]
    on_plus = on[on["signed_value"] > 0]["signed_value"].iloc[0]
    assert on_plus > off_plus
