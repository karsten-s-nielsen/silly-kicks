import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import add_defensive_credit
from tests.tracking._defensive_credit_fixtures import frame_with_defender, one_action


def _pass_scene(fitted_xt):
    a = one_action(type_name="pass", result_name="fail", start_x=95.0, start_y=34.0, team_id=10, player_id=5)
    a["shot_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["cross_blocked"] = pd.array([pd.NA], dtype="boolean")
    a["shot_on_target_derived"] = pd.array([pd.NA], dtype="boolean")  # present -> no TF-48 fallback
    a["xg"] = [np.nan]
    f = frame_with_defender(defender_x=96.0, defender_y=34.0)
    return a, f


def test_aggregate_excludes_acting_team_passer_debit(fitted_xt):
    a, f = _pass_scene(fitted_xt)
    out = add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    # pressure_pass_fail emits +presser(team 20) and -passer(team 10). Aggregate is defending-scoped:
    # net reflects the DEFENDING credit only, not cancel to 0.
    assert out["defensive_credit_plus"].iloc[0] > 0
    assert out["defensive_credit_minus"].iloc[0] == 0.0  # -passer excluded (acting team)
    assert out["defensive_credit_net"].iloc[0] == pytest.approx(out["defensive_credit_plus"].iloc[0])
    assert out["n_defensive_credits"].iloc[0] >= 1


def test_aggregate_always_finite_no_credit_action(fitted_xt):
    a, f = _pass_scene(fitted_xt)
    f.loc[f["player_id"] == 900, "x"] = 60.0  # defender far -> no credit
    out = add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    assert out["defensive_credit_net"].iloc[0] == 0.0
    assert out["n_defensive_credits"].iloc[0] == 0


def test_aggregate_is_pure(fitted_xt):
    a, f = _pass_scene(fitted_xt)
    before = a.copy()
    add_defensive_credit(a, f, xg_column="xg", xt=fitted_xt)
    pd.testing.assert_frame_equal(a, before)  # input unmutated
