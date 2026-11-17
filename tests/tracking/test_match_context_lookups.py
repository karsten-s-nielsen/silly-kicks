"""The ghost score and phase lookups (R50, R51, R52, R56; action-time repair spec §4.11, §4.12, §4.15, §5.2)."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._ghost_gk import _build_phase_lookup, _build_score_lookup

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id


def _rows(rows: list[tuple]) -> pd.DataFrame:
    """(period, time, team, type, result[, time_source[, provenance]])."""
    out = []
    for i, r in enumerate(rows):
        period, t, team, ty, res = r[:5]
        d = {
            "game_id": "100",
            "action_id": i,
            "period_id": period,
            "time_seconds": float(t),
            "team_id": team,
            "type_id": _T[ty],
            "result_id": _R[res],
        }
        if len(r) > 5:
            d["time_source"] = r[5]
        if len(r) > 6:
            d["action_provenance"] = r[6]
        out.append(d)
    return pd.DataFrame(out)


# ---- R51: the period ------------------------------------------------------------------------------------------


def test_phase_window_stays_in_its_period():
    fn = _build_phase_lookup(_rows([(1, 100, 1, "freekick_crossed", "success")]))
    assert fn("100", 1, 105.0) == 1
    assert fn("100", 2, 105.0) == 0  # a first-half set piece does not reach a second-half frame


def test_score_counts_earlier_periods_and_same_period_from_its_time():
    acts = _rows([(1, 600, 1, "shot", "success"), (2, 100, 2, "shot", "success")])
    fn = _build_score_lookup(acts, home_team_id=1)
    assert fn("100", 1, 300.0) == 0.0
    assert fn("100", 1, 700.0) == 1.0
    assert fn("100", 2, 50.0) == 1.0  # the first-half goal counts in the second half at a smaller clock time
    assert fn("100", 2, 150.0) == 0.0


def test_score_extra_time_counts_periods_one_to_three_at_period_four():
    acts = _rows([(1, 10, 1, "shot", "success"), (3, 60, 1, "shot", "success")])
    fn = _build_score_lookup(acts, home_team_id=1)
    assert fn("100", 4, 10.0) == 2.0


# ---- R52: the goal rule -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ty", "res"), [("shot_penalty", "success"), ("shot_freekick", "success"), ("bad_touch", "owngoal")]
)
def test_every_goal_of_the_library_rule_moves_the_score(ty, res):
    fn = _build_score_lookup(_rows([(1, 100, 1, ty, res), (1, 101, 2, "pass", "success")]), home_team_id=1)
    expected = -1.0 if res == "owngoal" else 1.0  # an own goal counts for the other team
    assert fn("100", 1, 200.0) == expected


def test_a_shoot_out_row_is_not_a_goal():
    acts = _rows([(1, 10, 1, "shot", "success"), (5, 30, 2, "shot_penalty", "success")])
    fn = _build_score_lookup(acts, home_team_id=1)
    assert fn("100", 5, 100.0) == 1.0  # a period-5 frame takes the score at the end of period 4


# ---- R50: an order-only row opens no window and ends none ----------------------------------------------------


@pytest.mark.parametrize("follower", ["throw_in", "pass", "shot_penalty"])
def test_order_only_row_opens_and_ends_no_window(follower):
    acts = _rows([
        (1, 10.0, 1, "freekick_crossed", "success", "placeholder", "derived"),
        (1, 14.0, 1, "corner_crossed", "fail", "order_only", "derived"),
        (1, 14.01, 2, follower, "success", "placeholder" if follower != "pass" else "tracking",
         "derived" if follower != "pass" else "native"),
    ])  # fmt: skip
    fn = _build_phase_lookup(acts)
    assert fn("100", 1, 15.0) == 1  # still the free kick's window (opened at 10 s), not the corner's
    assert fn("100", 1, 21.0) == 0  # the free kick's window has decayed; the order-only corner opened none


def test_placeholder_row_opens_a_window_as_today():
    fn = _build_phase_lookup(_rows([(1, 50.0, 1, "goalkick", "success", "placeholder", "derived")]))
    assert fn("100", 1, 55.0) == 2


def test_table_without_time_source_gives_todays_phase():
    fn = _build_phase_lookup(_rows([(1, 50.0, 1, "corner_crossed", "success")]))
    assert fn("100", 1, 55.0) == 1


# ---- R56: the SkillCorner running score ------------------------------------------------------------------


def _sc(rows: list[tuple]) -> pd.DataFrame:
    """(period, time, team, provenance, team_score, opponent_team_score)."""
    out = []
    for i, (period, t, team, prov, ts, os_) in enumerate(rows):
        out.append(
            {
                "game_id": "100",
                "action_id": i,
                "period_id": period,
                "time_seconds": float(t),
                "team_id": team,
                "type_id": _T["pass"],
                "result_id": _R["success"],
                "action_provenance": prov,
                "time_source": "tracking",
                "team_score": ts,
                "opponent_team_score": os_,
            }
        )
    return pd.DataFrame(out)


def test_running_score_takes_effect_at_the_next_native_row():
    acts = _sc([
        (1, 10, "363", "native", 0, 0),
        (1, 20, "363", "native", 1, 0),  # the higher score first shows here (the scoring possession)
        (1, 90, "400", "native", 0, 1),  # the kick-off possession: the effect is at its row time
    ])  # fmt: skip
    fn = _build_score_lookup(acts, home_team_id=363.0)  # type: ignore[arg-type]  # mixed id dtypes: 363.0 vs "363" (ids_match)
    assert fn("100", 1, 50.0) == 0.0  # no frame before the effect takes the new score
    assert fn("100", 1, 90.0) == 1.0


def test_running_score_period_5_frame_takes_the_score_at_the_end_of_period_4():
    # RTR-IMPL-01 (A01/B01): the R52 shoot-out rule applies to both paths (spec 4.12, 4.15); a period-5 native row
    # whose running score rises is a shoot-out kick, not a goal of the score
    acts = _sc([
        (4, 10, "363", "native", 1, 1),
        (4, 50, "400", "native", 1, 1),  # 1-1 at the end of period 4
        (5, 10, "363", "native", 2, 1),  # shoot-out kick scored: the provider score rises
        (5, 30, "400", "native", 1, 2),
        (5, 40, "363", "native", 2, 1),
    ])  # fmt: skip
    goal_path = _build_score_lookup(acts.drop(columns=["team_score", "opponent_team_score"]), home_team_id="363")
    fn = _build_score_lookup(acts, home_team_id="363")
    assert fn("100", 4, 200.0) == 0.0
    assert fn("100", 5, 35.0) == 0.0  # the end of period 4, as the goal-row path
    assert fn("100", 5, 35.0) == goal_path("100", 5, 35.0)


def test_running_score_period_5_frame_keeps_a_goal_on_the_last_row_of_period_4():
    # the score at the end of period 4 is the score the last native row of period 4 shows, even when that row (the
    # scoring possession) has no next row in period 4 to carry the effect
    acts = _sc([
        (4, 10, "363", "native", 0, 0),
        (4, 50, "363", "native", 1, 0),  # a late goal: the scoring possession is the last row of period 4
        (5, 10, "400", "native", 2, 1),  # shoot-out
    ])  # fmt: skip
    fn = _build_score_lookup(acts, home_team_id="363")
    assert fn("100", 4, 60.0) == 0.0  # in period 4 the effect rule holds: no next row, no effect
    assert fn("100", 5, 20.0) == 1.0  # a period-5 frame takes the score at the end of period 4


def test_running_score_ignores_derived_rows():
    acts = _sc([
        (1, 10, "363", "native", 0, 0),
        (1, 15, "400", "derived", 9, 9),
        (1, 20, "363", "native", 0, 0),
    ])  # fmt: skip
    fn = _build_score_lookup(acts, home_team_id="363")
    assert fn("100", 1, 30.0) == 0.0


@pytest.mark.parametrize("second", [(0, 0), (None, 0)], ids=["decrease", "partial-na"])
def test_an_inconsistent_running_score_falls_back_to_the_goal_rows(second):
    acts = _sc([(1, 10, "363", "native", 1, 0), (1, 20, "363", "native", *second), (1, 30, "400", "native", 0, 0)])
    acts.loc[0, "type_id"] = _T["shot"]  # one goal row by the R52 rule
    fn = _build_score_lookup(acts, home_team_id="363")
    assert fn("100", 1, 40.0) == 1.0  # the goal row, not the running score


def test_no_running_score_columns_use_the_goal_rows():
    acts = _rows([(1, 100, 1, "shot", "success")])
    assert _build_score_lookup(acts, home_team_id=1)("100", 1, 150.0) == 1.0


def test_phase_extra_time_window_stays_in_period_three():
    fn = _build_phase_lookup(_rows([(3, 890, 1, "corner_short", "success")]))
    assert fn("100", 3, 895.0) == 1
    assert fn("100", 4, 5.0) == 0


def test_order_only_row_with_a_goal_follower_opens_no_window():
    # a goal follows the restart gap: no follower row
    acts = _rows([(1, 30.0, 1, "corner_crossed", "fail", "order_only", "derived")])
    assert _build_phase_lookup(acts)("100", 1, 31.0) == 0


# ---- each caller passes the period (spec section 4.11) ---------------------------------------------------


def _recorder(seen: list, value: float = 0.0):
    def fn(game_id, period_id, time_s) -> Any:
        seen.append(int(period_id))
        return value

    return fn


def test_ghost_gk_extraction_passes_the_frame_period():
    from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features
    from tests.tracking.test_ghost_gk import _make_multi_frame_fixture

    frames = _make_multi_frame_fixture(n_frames=1).assign(period_id=2)
    score, phase = [], []
    _extract_all_ghost_gk_features(frames, home_team_id=1, score_at_time=_recorder(score),
                                   phase_at_time=_recorder(phase))  # fmt: skip
    assert score and set(score) == {2} and set(phase) == {2}


def test_ghost_outfield_extraction_passes_the_frame_period(monkeypatch):
    import silly_kicks.tracking._ghost_outfield as go

    frame = go._outfield_probe_frame().assign(period_id=2)
    score, phase = [], []
    monkeypatch.setattr(go, "_build_score_lookup", lambda *a, **k: _recorder(score))
    monkeypatch.setattr(go, "_build_phase_lookup", lambda *a, **k: _recorder(phase))
    home = frame["team_id"].dropna().iloc[0]
    go._extract_all_ghost_outfield_features(frame, go._outfield_probe_actions(), home_team_id=home)
    assert score and set(score) == {2} and set(phase) == {2}


def _patch_score_lookup(monkeypatch, seen):
    import silly_kicks.tracking._ghost_gk as gg

    monkeypatch.setattr(gg, "_build_score_lookup", lambda *a, **k: _recorder(seen))


def _xcross_actions(actions):
    goal = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [0.0],
                         "type_id": [_T["shot"]], "result_id": [_R["success"]]})  # fmt: skip
    return pd.concat([actions.assign(result_id=_R["success"]), goal], ignore_index=True)


def test_xcross_training_data_passes_the_period(monkeypatch):
    import silly_kicks.tracking._xcross_attempt as xc
    from tests.tracking.test_xcross_attempt import _mini_match

    frames, actions = _mini_match()
    seen: list = []
    _patch_score_lookup(monkeypatch, seen)
    xc.prepare_xcross_training_data(frames, _xcross_actions(actions), home_team_id="A")
    assert seen and set(seen) == set(frames["period_id"].astype(int))


def test_xcross_serve_passes_the_period(monkeypatch):
    import silly_kicks.tracking._xcross_attempt as xc
    from tests.tracking.test_xcross_attempt import _fit_on_mini

    frames, actions, model = _fit_on_mini()
    seen: list = []
    _patch_score_lookup(monkeypatch, seen)
    xc.compute_xcross_attempt(frames, model=model, home_team_id="A", actions=_xcross_actions(actions))
    assert seen and set(seen) == set(frames["period_id"].astype(int))


def test_causal_opportunities_pass_the_period(monkeypatch):
    from silly_kicks.causal import opportunities as O
    from tests.causal._fixtures import META, actions, spell

    f = spell(5, 10.0, 11.0, period=2)
    a = actions([[1, 0, 2, 5, 10.5, _T["cross"], 1, 20, 8, 14, 6]])
    seen: list = []
    _patch_score_lookup(monkeypatch, seen)
    O.build_opportunities(f, a, home_team_id=5, model_metadata=META)
    assert seen and set(seen) == {2}
