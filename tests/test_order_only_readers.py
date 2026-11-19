"""The action-window readers and an ``"order_only"`` row (action-time repair spec section 4.13).

Each reader takes no window at an ``"order_only"`` row (a restart with no first possession: its time keeps the row
order only). On a table with no such row, or without ``time_source``, each gives today's values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id


def _a(rows: list[tuple], *, with_source: bool = True) -> pd.DataFrame:
    """(t, team, type, result[, time_source]) -> a one-game, one-period SPADL table."""
    out = []
    for i, r in enumerate(rows):
        t, team, ty, res = r[:4]
        src = r[4] if len(r) > 4 else "event"
        d = {
            "game_id": 1,
            "period_id": 1,
            "action_id": i,
            "time_seconds": float(t),
            "team_id": team,
            "player_id": team * 10 + i,
            "type_id": _T[ty],
            "result_id": _R[res],
            "bodypart_id": 0,
            "start_x": 60.0,
            "start_y": 34.0,
            "end_x": 70.0,
            "end_y": 34.0,
            "action_provenance": "derived" if src in ("order_only", "placeholder") else "native",
        }
        if with_source:
            d["time_source"] = src
        out.append(d)
    return pd.DataFrame(out)


# ---- VAEP scores / concedes, window="time" --------------------------------------------------------------


def _vaep_rows(corner_src: str) -> list[tuple]:
    return [(10.0, 1, "pass", "success"), (11.0, 1, "corner_crossed", "success", corner_src),
            (12.0, 1, "shot", "success")]  # fmt: skip


def test_vaep_time_window_order_only_row_is_na():
    from silly_kicks.vaep.labels import concedes, scores

    a = _a(_vaep_rows("order_only"))
    s = scores(a, window="time", window_seconds=5.0)["scores"]
    assert str(s.dtype) == "boolean"
    assert s.tolist()[0] is True and s.tolist()[2] is True and pd.isna(s.iloc[1])
    assert pd.isna(concedes(a, window="time", window_seconds=5.0)["concedes"].iloc[1])


def test_vaep_time_window_xg_order_only_row_is_nan():
    from silly_kicks.vaep.labels import scores

    a = _a(_vaep_rows("order_only")).assign(xg=[0.0, 0.0, 0.3])
    s = scores(a, window="time", window_seconds=5.0, xg_column="xg")["scores"]
    assert s.dtype == np.float64 and np.isnan(s.iloc[1]) and s.iloc[0] == pytest.approx(0.3)


def test_vaep_time_window_order_only_penalty_is_no_goal():
    from silly_kicks.vaep.labels import concedes

    a = _a([(10.0, 1, "pass", "success"), (11.0, 2, "shot_penalty", "success", "order_only")])
    c = concedes(a, window="time", window_seconds=5.0)["concedes"]
    assert pd.notna(c.iloc[0]) and not bool(c.iloc[0])  # the order-only row is no goal for the other team


@pytest.mark.parametrize("with_source", [True, False])
def test_vaep_time_window_without_order_only_gives_todays_values(with_source):
    from silly_kicks.vaep.labels import concedes, scores

    a = _a(_vaep_rows("placeholder"), with_source=with_source)
    s = scores(a, window="time", window_seconds=5.0)["scores"]
    c = concedes(a, window="time", window_seconds=5.0)["concedes"]
    assert str(s.dtype) == "boolean" and str(c.dtype) == "boolean"  # the dtype does not depend on the data
    assert s.tolist() == [True, True, True] and c.tolist() == [False, False, False]


# ---- retains --------------------------------------------------------------------------------------------


def test_retains_order_only_anchor_is_nan():
    from silly_kicks.xtgk._retention_labels import retains

    rows = [(10.0, 1, "pass", "success"), (11.0, 1, "corner_crossed", "success", "order_only"),
            (12.0, 1, "shot", "fail")]  # fmt: skip
    r = retains(_a(rows))
    assert np.isnan(r.iloc[1]) and r.iloc[0] == 1.0


def test_retains_without_order_only_gives_todays_values():
    from silly_kicks.xtgk._retention_labels import retains

    rows = [(10.0, 1, "pass", "success"), (11.0, 1, "corner_crossed", "success", "placeholder"),
            (12.0, 1, "shot", "fail")]  # fmt: skip
    pd.testing.assert_series_equal(retains(_a(rows)), retains(_a(rows, with_source=False)))
    assert retains(_a(rows)).iloc[1] == 1.0


# ---- EmpiricalTurnoverValue, finite window_seconds -------------------------------------------------------


def _turnover_table(src: str) -> pd.DataFrame:
    # a failed goal kick is a turnover anchor of the move set (a corner is not)
    a = _a([(0.0, 10, "goalkick", "fail", src), (5.0, 20, "shot", "fail")])
    a["start_x"] = [5.0, 100.0]
    a["possession_id"] = [0, 1]
    a["xg"] = [np.nan, 0.4]
    a["pressure"] = 0.1
    return a


@pytest.mark.parametrize(("src", "support"), [("order_only", 0), ("placeholder", 1)])
def test_turnover_window_order_only_row_is_no_anchor(src, support):
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    etv = EmpiricalTurnoverValue(window_seconds=10.0, min_support=1).fit(
        _turnover_table(src), xg_column="xg", pressure_column="pressure"
    )
    assert int(sum(np.asarray(etv.support(p)).sum() for p in (1, 2, 3))) == support


def test_turnover_possession_bound_scope_is_unchanged():
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue

    etv = EmpiricalTurnoverValue(window_seconds=None, min_support=1).fit(
        _turnover_table("order_only"), xg_column="xg", pressure_column="pressure"
    )
    assert int(sum(np.asarray(etv.support(p)).sum() for p in (1, 2, 3))) == 1


# ---- TF-49 packing, the secured window ----------------------------------------------------------------


@pytest.mark.parametrize(("src", "is_na"), [("order_only", True), ("placeholder", False)])
def test_packing_secured_is_na_when_the_reception_row_is_order_only(src, is_na):
    from silly_kicks.tracking._packing import secured_reception

    a = _a([(10.0, 1, "pass", "success"), (11.0, 1, "corner_crossed", "success", src),
            (12.0, 1, "shot", "fail")])  # fmt: skip
    line_x = pd.Series([30.0, np.nan, np.nan])
    receiver = pd.Series([1, pd.NA, pd.NA], dtype="Int64")
    out = secured_reception(a, line_x, receiver)
    assert pd.isna(out.iloc[0]) is is_na


# ---- team metrics: spells ------------------------------------------------------------------------------


def _ctx(srcs: list[str]) -> pd.DataFrame:
    a = _a([(10.0, 1, "corner_crossed", "success", srcs[0]), (12.0, 1, "pass", "success", srcs[1]),
            (14.0, 2, "throw_in", "success", srcs[2]), (20.0, 1, "pass", "success", srcs[3])])  # fmt: skip
    a["possession_id"] = [0, 0, 1, 2]
    a["team_in_possession"] = a["team_id"]
    return a


def test_spell_times_skip_order_only_rows():
    from silly_kicks.team_metrics._possession import build_spells

    s = build_spells(_ctx(["order_only", "event", "order_only", "event"]))
    first = s[s["possession_id"] == 0].iloc[0]
    assert first["start_time"] == 12.0 and first["end_time"] == 12.0
    only_oo = s[s["possession_id"] == 1].iloc[0]  # a spell of order_only rows only
    assert np.isnan(only_oo["start_time"]) and np.isnan(only_oo["duration_s"]) and not bool(only_oo["is_recovery"])


def test_spells_without_order_only_give_todays_values():
    from silly_kicks.team_metrics._possession import build_spells

    ctx = _ctx(["placeholder", "event", "placeholder", "event"])
    pd.testing.assert_frame_equal(build_spells(ctx), build_spells(ctx.drop(columns=["time_source"])))
    assert build_spells(ctx)["start_time"].tolist() == [10.0, 14.0, 20.0]


# ---- the possession carve-out brief_window_seconds ------------------------------------------------------


def _brief(src: str) -> pd.DataFrame:
    return _a([(10.0, 1, "pass", "success"), (11.0, 2, "corner_crossed", "fail", src),
               (12.0, 1, "pass", "success")])  # fmt: skip


@pytest.mark.parametrize(("src", "n_poss"), [("order_only", 3), ("placeholder", 1)])
def test_brief_carve_out_reads_no_order_only_row(src, n_poss):
    from silly_kicks.spadl.utils import add_possessions

    out = add_possessions(_brief(src), merge_brief_opposing_actions=1, brief_window_seconds=2.0)
    assert out["possession_id"].nunique() == n_poss


@pytest.mark.parametrize(("src", "n_poss"), [("order_only", 3), ("placeholder", 1)])
def test_atomic_brief_carve_out_reads_no_order_only_row(src, n_poss):
    from silly_kicks.atomic.spadl.utils import add_possessions

    a = _brief(src).rename(columns={"start_x": "x", "start_y": "y"})
    a["dx"], a["dy"] = 10.0, 0.0
    a = a.drop(columns=["end_x", "end_y", "result_id"])
    a["type_id"] = [0, 0, 0]  # atomic pass for every row; the order-only flag decides
    out = add_possessions(a, merge_brief_opposing_actions=1, brief_window_seconds=2.0)
    assert out["possession_id"].nunique() == n_poss


# ---- coordination windows from actions --------------------------------------------------------------


def test_event_intervals_skip_order_only_rows():
    from silly_kicks.coordination._windows import _event_intervals

    a = _a([(10.0, 1, "pass", "success"), (11.0, 2, "corner_crossed", "fail", "order_only"),
            (15.0, 1, "throw_in", "success", "placeholder"), (20.0, 1, "pass", "success")])  # fmt: skip
    iv = _event_intervals(a, 0.0)[(1, 1)].tolist()
    # the order-only corner: previous other row (10 s) to next other row (15 s); the throw-in: 10 s (not 11 s) to 15 s
    assert set(map(tuple, iv)) == {(10.0, 15.0)}


def test_event_intervals_open_no_goal_interval_at_an_unmeasured_shot():
    from silly_kicks.coordination._windows import _event_intervals

    a = _a([(10.0, 1, "shot_penalty", "success", "placeholder"), (20.0, 2, "pass", "success")])
    assert _event_intervals(a, 0.0) == {}


def test_event_intervals_without_order_only_give_todays_values():
    from silly_kicks.coordination._windows import _event_intervals

    a = _a([(10.0, 1, "pass", "success"), (15.0, 1, "throw_in", "success", "event"), (20.0, 1, "shot", "success"),
            (30.0, 2, "pass", "success")])  # fmt: skip
    got = _event_intervals(a, 0.0)
    want = _event_intervals(a.drop(columns=["time_source"]), 0.0)
    assert {k: v.tolist() for k, v in got.items()} == {k: v.tolist() for k, v in want.items()}
    assert sorted(map(tuple, got[(1, 1)].tolist())) == [(10.0, 15.0), (20.0, 30.0)]


def test_possession_windows_do_not_start_at_an_order_only_row():
    from silly_kicks.coordination._windows import possession_windows_from_actions
    from tests.coordination._fixtures import make_coordination_match

    frames = make_coordination_match(seconds=60.0, hz=10.0)
    a = _a([(5.0, 1, "pass", "success"), (10.0, 2, "corner_crossed", "fail", "order_only"),
            (12.0, 2, "pass", "success"), (20.0, 1, "tackle", "success")])  # fmt: skip
    a["team_id"] = a["team_id"].astype("int64")
    w = possession_windows_from_actions(a, frames).sort_values("start_time_s").reset_index(drop=True)
    assert 10.0 not in w["start_time_s"].tolist() and 12.0 in w["start_time_s"].tolist()
    assert w.loc[w["start_time_s"] == 5.0, "end_time_s"].iloc[0] == 12.0


def test_possession_windows_without_order_only_give_todays_values():
    from silly_kicks.coordination._windows import possession_windows_from_actions
    from tests.coordination._fixtures import make_coordination_match

    frames = make_coordination_match(seconds=60.0, hz=10.0)
    a = _a([(5.0, 1, "pass", "success"), (10.0, 2, "corner_crossed", "fail", "placeholder"),
            (12.0, 2, "pass", "success"), (20.0, 1, "tackle", "success")])  # fmt: skip
    pd.testing.assert_frame_equal(
        possession_windows_from_actions(a, frames),
        possession_windows_from_actions(a.drop(columns=["time_source"]), frames),
    )
