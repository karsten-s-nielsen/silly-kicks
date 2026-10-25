import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import MarkovPossessionValue, PressureLevels
from silly_kicks.xtgk._possession_value import mirror_zone, zone_of
from silly_kicks.xthreat._grid import M, N  # M=12 (w), N=16 (l)
from tests.xtgk.conftest import three_band_cohort


def test_mirror_zone_is_an_involution():
    for z in range(M * N):
        assert mirror_zone(mirror_zone(z, N, M), N, M) == z


def test_mirror_zone_reflects_both_axes():
    # deep-left-bottom cell -> attacking-right-top cell (180 deg point reflection)
    z_deep = zone_of(3.0, 4.0, N, M)  # xi~0, low y
    z_far = mirror_zone(z_deep, N, M)
    xi, yj = z_far % N, z_far // N
    assert xi == N - 1 - (z_deep % N)  # column reversed
    assert yj == M - 1 - (z_deep // N)  # row reversed


def _fit_v():
    actions = three_band_cohort()
    pl = PressureLevels().fit(actions["pressure"])
    return MarkovPossessionValue().fit(actions, xg_column="xg", pressure_column="pressure", pressure_levels=pl)


# ---------------------------------------------------------------------------
# TF-46 Arm 1 (decision B5/E, TF46C-IMPL-20): the xT-GK fits exclude the shared-helper rows.
# ---------------------------------------------------------------------------
def _k2_marked_rows(n: int = 15) -> pd.DataFrame:
    # Extra goal-kick/FAIL rows marked result_source="inferred" (the K2 kept set-piece rows). Without the
    # exclusion they enter the move-set and change the empirical surfaces; the helper must drop them.
    extra = three_band_cohort().head(n).copy()
    extra["type_id"] = spadlconfig.actiontype_id["goalkick"]
    extra["result_id"] = spadlconfig.result_id["fail"]
    extra["result_source"] = "inferred"
    return extra


def _assert_surfaces_equal(a, b) -> None:
    assert set(a._surfaces) == set(b._surfaces)
    for p in a._surfaces:
        np.testing.assert_allclose(a.surface(p), b.surface(p))


def test_markov_fit_excludes_training_marked_rows():
    base = three_band_cohort()
    pl = PressureLevels().fit(base["pressure"])
    m_base = MarkovPossessionValue().fit(base, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    combined = pd.concat([base, _k2_marked_rows()], ignore_index=True)
    m_comb = MarkovPossessionValue().fit(combined, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    for p in m_base._surfaces:
        np.testing.assert_allclose(m_base._surfaces[p], m_comb._surfaces[p])


def test_markov_fit_tolerates_absent_columns():
    base = three_band_cohort()  # no result_source column
    with_na = base.copy()
    with_na["result_source"] = pd.NA
    pl = PressureLevels().fit(base["pressure"])
    a = MarkovPossessionValue().fit(base, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    b = MarkovPossessionValue().fit(with_na, xg_column="xg", pressure_column="pressure", pressure_levels=pl)
    for p in a._surfaces:
        np.testing.assert_allclose(a._surfaces[p], b._surfaces[p])


def test_turnover_fit_excludes_training_marked_rows():
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    base = three_band_cohort()
    t_base = EmpiricalTurnoverValue(min_support=1).fit(base, xg_column="xg", pressure_column="pressure")
    combined = pd.concat([base, _k2_marked_rows()], ignore_index=True)
    t_comb = EmpiricalTurnoverValue(min_support=1).fit(combined, xg_column="xg", pressure_column="pressure")
    _assert_surfaces_equal(t_base, t_comb)


def test_turnover_fit_tolerates_absent_columns():
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    base = three_band_cohort()  # no result_source column
    with_na = base.copy()
    with_na["result_source"] = pd.NA
    t_base = EmpiricalTurnoverValue(min_support=1).fit(base, xg_column="xg", pressure_column="pressure")
    t_na = EmpiricalTurnoverValue(min_support=1).fit(with_na, xg_column="xg", pressure_column="pressure")
    _assert_surfaces_equal(t_base, t_na)


def test_turnover_keeps_marked_shot_as_context():
    # TF46C-IMPL-25: a repair_kind SHOT (a reconciled goal) is excluded as a turnover EXAMPLE but KEPT as
    # context -- the opponent-post-turnover-shot credit must still see it. Dropping the row (the rejected fix)
    # would give the loss zone zero credit.
    from silly_kicks.xtgk._possession_value import zone_of
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    PASS = spadlconfig.actiontype_id["pass"]
    SHOT = spadlconfig.actiontype_id["shot"]
    FAIL = spadlconfig.result_id["fail"]
    rows = [
        dict(
            game_id=1,
            period_id=1,
            action_id=0,
            time_seconds=0.0,
            team_id=10,
            player_id=1,
            type_id=PASS,
            result_id=FAIL,
            bodypart_id=0,
            start_x=5.0,
            start_y=34.0,
            end_x=20.0,
            end_y=34.0,
            possession_id=1,
            xg=np.nan,
            pressure=0.1,
            repair_kind=pd.NA,
        ),
        dict(
            game_id=1,
            period_id=1,
            action_id=1,
            time_seconds=2.0,
            team_id=20,
            player_id=2,
            type_id=SHOT,
            result_id=FAIL,
            bodypart_id=0,
            start_x=100.0,
            start_y=34.0,
            end_x=105.0,
            end_y=34.0,
            possession_id=2,
            xg=0.4,
            pressure=0.1,
            repair_kind="own_goal_inverted",
        ),
    ]
    etv = EmpiricalTurnoverValue(min_support=1).fit(pd.DataFrame(rows), xg_column="xg", pressure_column="pressure")
    z_loss = zone_of(5.0, 34.0)
    # Both rows are pressure tercile 1; the marked opponent shot is still credited to the loss zone (dropping
    # it -- the rejected r2 behaviour -- would give 0). Literal level keeps the PressureLevel type (pyright).
    assert etv.value(z_loss, 1) > 0.0


def test_mirrored_turnover_equals_v_at_the_mirror_zone():
    from silly_kicks.xtgk._turnover import MirroredTurnoverCost, TurnoverCost

    v = _fit_v()
    tc = MirroredTurnoverCost(v)
    assert isinstance(tc, TurnoverCost)
    for z in (0, 1, 20, 100):
        assert tc.value(z, 1) == v.value(mirror_zone(z), 1)


def test_pressure_transfer_policy_is_injectable():
    from silly_kicks.xtgk._turnover import MirroredTurnoverCost

    v = _fit_v()
    tc = MirroredTurnoverCost(v, pressure_policy=lambda p: 1)  # opponent always low pressure
    assert tc.value(0, 3) == v.value(mirror_zone(0), 1)


def test_mirrored_surface_and_support_are_point_reflections():
    from silly_kicks.xtgk._turnover import MirroredTurnoverCost

    v = _fit_v()
    tc = MirroredTurnoverCost(v)
    assert np.array_equal(tc.surface(1), np.asarray(v.surface(1))[::-1, ::-1])
    assert np.array_equal(tc.support(1), np.asarray(v.support(1))[::-1, ::-1])


def test_empirical_turnover_credits_the_opponents_post_turnover_shot():
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    PASS = spadlconfig.actiontype_id["pass"]
    SHOT = spadlconfig.actiontype_id["shot"]
    SUCCESS = spadlconfig.result_id["success"]
    FAIL = spadlconfig.result_id["fail"]
    rows = [
        dict(
            game_id=1,
            period_id=1,
            action_id=0,
            time_seconds=0.0,
            team_id=10,
            player_id=1,
            type_id=PASS,
            result_id=FAIL,
            bodypart_id=0,
            start_x=5.0,
            start_y=34.0,
            end_x=20.0,
            end_y=34.0,
            possession_id=0,
            xg=np.nan,
            pressure=0.1,
        ),
        dict(
            game_id=1,
            period_id=1,
            action_id=1,
            time_seconds=1.0,
            team_id=20,
            player_id=2,
            type_id=PASS,
            result_id=SUCCESS,
            bodypart_id=0,
            start_x=85.0,
            start_y=34.0,
            end_x=100.0,
            end_y=34.0,
            possession_id=1,
            xg=np.nan,
            pressure=0.1,
        ),
        dict(
            game_id=1,
            period_id=1,
            action_id=2,
            time_seconds=2.0,
            team_id=20,
            player_id=2,
            type_id=SHOT,
            result_id=FAIL,
            bodypart_id=0,
            start_x=100.0,
            start_y=34.0,
            end_x=105.0,
            end_y=34.0,
            possession_id=1,
            xg=0.4,
            pressure=0.1,
        ),
    ]
    actions = pd.DataFrame(rows)
    etv = EmpiricalTurnoverValue(min_support=1).fit(actions, xg_column="xg", pressure_column="pressure")
    z_loss = zone_of(5.0, 34.0)
    assert etv.value(z_loss, 1) > 0.0  # opponent's post-turnover shot xg credited to the loss zone


def test_empirical_turnover_ignores_shot_outside_time_window():
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    PASS = spadlconfig.actiontype_id["pass"]
    SHOT = spadlconfig.actiontype_id["shot"]
    FAIL = spadlconfig.result_id["fail"]
    rows = [
        dict(
            game_id=1,
            period_id=1,
            action_id=0,
            time_seconds=0.0,
            team_id=10,
            player_id=1,
            type_id=PASS,
            result_id=FAIL,
            bodypart_id=0,
            start_x=5.0,
            start_y=34.0,
            end_x=20.0,
            end_y=34.0,
            possession_id=0,
            xg=np.nan,
            pressure=0.1,
        ),
        dict(
            game_id=1,
            period_id=1,
            action_id=1,
            time_seconds=99.0,
            team_id=20,
            player_id=2,
            type_id=SHOT,
            result_id=FAIL,
            bodypart_id=0,
            start_x=100.0,
            start_y=34.0,
            end_x=105.0,
            end_y=34.0,
            possession_id=1,
            xg=0.4,
            pressure=0.1,
        ),  # 99s later -> out of window
    ]
    actions = pd.DataFrame(rows)
    etv = EmpiricalTurnoverValue(window_seconds=10.0, min_support=1).fit(
        actions, xg_column="xg", pressure_column="pressure"
    )
    assert etv.value(zone_of(5.0, 34.0), 1) == 0.0


def test_deep_loss_has_higher_cost_than_final_third_loss():
    from silly_kicks.xtgk._turnover import MirroredTurnoverCost

    v = _fit_v()
    tc = MirroredTurnoverCost(v)
    z_deep = zone_of(3.0, 34.0)
    z_final = zone_of(100.0, 34.0)
    assert tc.value(z_deep, 1) >= tc.value(z_final, 1)
