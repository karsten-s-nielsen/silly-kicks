"""`derive_match_states` and `_score_states` over arrays equal their `2474b73` per-row bodies (HFU plan R3-8 S3
item 7 (a)).

On HF the two per-row `iterrows` loops took 2,060 of the 2,294 profiled seconds of the `win_probability` `fit`
(S1f). No-rebaseline kind (c): `_derive_oracle` and `_score_oracle` below are the `2474b73` bodies VERBATIM, and
every case compares the new output with the oracle's exactly: frame values, dtypes and the Python type of every
`team_id` value, the score arrays byte for byte, and the same exception when the oracle raises. The inputs cover
both row-typing regimes of `iterrows` (an all-numeric frame gives float rows; an object column gives object rows),
string, float-with-NaN and nullable team ids, null-actor rows with and without scoring events, one- and
three-team games, missing home teams, missing player ids, unknown periods and repeated yellow cards. The public
`compute_win_probability` and `goal_leverage` are compared end to end against the same call with the oracles
patched in.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import silly_kicks.win_probability._compute as compute_mod
from silly_kicks.id_compat import canonical_id_series, same_id
from silly_kicks.spadl import config as _spadl
from silly_kicks.win_probability import (
    WinProbabilityModel,
    WinProbabilityParams,
    compute_win_probability,
    goal_leverage,
)
from silly_kicks.win_probability._compute import _score_states, _strength_map
from silly_kicks.win_probability._state import _home_team_map, derive_match_states

_P = WinProbabilityParams.default()
_SHOT_TYPE_IDS = [_spadl.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_SUCCESS = _spadl.result_id["success"]
_OWNGOAL = _spadl.result_id["owngoal"]
_RED = _spadl.result_id["red_card"]
_YELLOW = _spadl.result_id["yellow_card"]
_FAIL = _spadl.result_id["fail"]
_FOUL = _spadl.actiontype_id["foul"]
_PASS = _spadl.actiontype_id["pass"]
_PERIOD_OFFSET_MIN = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}


# The `2474b73` `derive_match_states`, VERBATIM (the D20 oracle; only the name differs).
def _derive_oracle(
    actions: pd.DataFrame,
    *,
    games: pd.DataFrame | None = None,
    params: WinProbabilityParams,
) -> pd.DataFrame:
    """Per-action PRE-action game state. Pure -- never mutates ``actions``."""
    home_map = _home_team_map(actions, games)
    a = actions.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)

    rows: list[dict] = []
    for game_id, g in a.groupby("game_id", sort=False):
        home_id = home_map.get(game_id)
        teams = list(pd.unique(g["team_id"].dropna()))
        goals = {t: 0 for t in teams}
        reds = {t: 0 for t in teams}
        yellows: dict = {t: {} for t in teams}
        scored_np = (g["type_id"].isin(_SHOT_TYPE_IDS) & (g["result_id"] == _SUCCESS)).to_numpy()
        og_np = (g["result_id"] == _OWNGOAL).to_numpy()
        period_np = g["period_id"].to_numpy()
        time_np = g["time_seconds"].to_numpy()
        last_period = int(period_np.max())
        last_time = float(time_np[period_np == last_period].max())
        final_min = _PERIOD_OFFSET_MIN.get(last_period, 90) + last_time / 60.0
        final_min = max(final_min, params.regulation_minutes)
        for j, (_, r) in enumerate(g.iterrows()):
            t = r["team_id"]
            opp_candidates = [x for x in teams if not same_id(x, t)]
            opp = opp_candidates[0] if len(opp_candidates) == 1 else None
            abs_min = _PERIOD_OFFSET_MIN.get(int(r["period_id"]), 0) + float(r["time_seconds"]) / 60.0
            own_s = goals.get(t, 0)
            opp_s = goals.get(opp, 0) if opp is not None else 0
            man = (reds.get(opp, 0) - reds.get(t, 0)) if opp is not None else 0
            rows.append(
                {
                    "game_id": game_id,
                    "action_id": int(r["action_id"]),
                    "period_id": int(r["period_id"]),
                    "team_id": t,
                    "absolute_minute": abs_min,
                    "minutes_remaining": max(final_min - abs_min, 0.0),
                    "own_score": own_s,
                    "opp_score": opp_s,
                    "score_diff": own_s - opp_s,
                    "home": (home_id is not None and same_id(t, home_id)),
                    "man_advantage": int(man),
                    "state_source": "resolved" if (home_id is not None and opp is not None) else "unresolved",
                }
            )
            # advance running tallies AFTER recording the pre-action state
            if bool(scored_np[j]):
                goals[t] = goals.get(t, 0) + 1
            elif bool(og_np[j]) and opp is not None:
                goals[opp] = goals.get(opp, 0) + 1
            if r["type_id"] == _FOUL and r["result_id"] == _RED:
                reds[t] = reds.get(t, 0) + 1
            elif r["type_id"] == _FOUL and r["result_id"] == _YELLOW:
                pid = r.get("player_id")
                if pid is not None and not pd.isna(pid):
                    yellows[t][pid] = yellows[t].get(pid, 0) + 1
                    if yellows[t][pid] >= 2:
                        reds[t] = reds.get(t, 0) + 1  # second yellow -> red
    return pd.DataFrame(rows).astype({"man_advantage": "int64", "home": "boolean"})


# The `2474b73` `_score_states`, VERBATIM (the D20 oracle; only the name differs).
def _score_oracle(states: pd.DataFrame, model: WinProbabilityModel, strength: dict, params: WinProbabilityParams):
    """Return per-state (p_win, p_draw, p_loss, leverage) via one table per (game, team, man_advantage)."""
    K = params.lattice_pad
    step = params.interval_minutes
    n = len(states)
    pw = np.full(n, np.nan)
    pd_ = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    lev = np.full(n, np.nan)
    cg = canonical_id_series(states["game_id"]).to_numpy()
    grp = states.assign(_cg=cg).groupby(["_cg", "team_id", "man_advantage"], sort=False)
    for (g, t, _man_adv_key), sub in grp:
        home = bool(sub["home"].iloc[0])
        bs = strength.get((g, t), 0.0)
        man_adv = int(sub["man_advantage"].to_numpy()[0])
        max_rem = float(sub["minutes_remaining"].max())
        n_tab = max(math.ceil(max_rem / step), 0)
        Pw, Pd, Pl = model.outcome_table(base_strength=bs, home=home, man_advantage=man_adv, n_intervals=n_tab)
        for pos, (_, r) in zip(sub["_pos"].to_numpy(), sub.iterrows(), strict=False):
            m = max(math.ceil(float(r["minutes_remaining"]) / step), 0)
            di = int(np.clip(int(r["score_diff"]) + K, 0, 2 * K))  # clamp extreme blowouts to the pad
            w_raw, d_raw, l_raw = float(Pw[di, m]), float(Pd[di, m]), float(Pl[di, m])
            w = model._apply_isotonic(w_raw)
            rem = max(1.0 - w, 0.0)
            dl = d_raw + l_raw
            d_cal = rem * (d_raw / dl) if dl > 0 else rem / 2.0
            di_up = min(di + 1, 2 * K)
            leverage = model._apply_isotonic(float(Pw[di_up, m])) - w
            pw[pos] = w
            pd_[pos] = d_cal
            pl[pos] = rem - d_cal
            lev[pos] = leverage
    return pw, pd_, pl, lev


# --------------------------------------------------------------------------------------------- inputs


def _random_actions(
    seed: int,
    *,
    team_kind: str = "int",
    object_column: bool = True,
    null_actor_events: bool = False,
    player_column: bool = True,
    n_games: int = 5,
    n_per_game: int = 220,
) -> pd.DataFrame:
    """Matches with shots, own goals, red cards and repeated yellows; some null-actor rows."""
    rng = np.random.default_rng(seed)
    frames = []
    for gi in range(n_games):
        n_teams = 1 if gi == 3 else 3 if gi == 4 else 2
        if team_kind == "str":
            teams = [f"T{gi}{k}" for k in range(n_teams)]
        else:
            teams = [100 + 10 * gi + k for k in range(n_teams)]
        n = n_per_game
        period = rng.choice([1, 2, 3, 4, 6] if gi == 2 else [1, 2], n, p=[0.3, 0.3, 0.2, 0.1, 0.1] if gi == 2 else None)
        kinds = rng.choice(5, n, p=[0.70, 0.12, 0.03, 0.10, 0.05])  # pass, shot, own goal, yellow foul, red foul
        type_id = np.where(kinds == 1, rng.choice(_SHOT_TYPE_IDS, n), np.where(kinds >= 3, _FOUL, _PASS))
        result_id = np.select(
            [kinds == 1, kinds == 2, kinds == 3, kinds == 4],
            [np.where(rng.random(n) < 0.3, _SUCCESS, _FAIL), _OWNGOAL, _YELLOW, _RED],
            _SUCCESS,
        )
        team = np.asarray(teams, dtype=object)[rng.integers(0, n_teams, n)]
        frame = pd.DataFrame(
            {
                "game_id": gi if team_kind == "int" and not object_column else f"game{gi}",
                "action_id": rng.permutation(n),
                "period_id": period,
                "time_seconds": rng.uniform(0.0, 2900.0, n).round(1),
                "team_id": list(team),
                "type_id": type_id,
                "result_id": result_id,
            }
        )
        if player_column:
            frame["player_id"] = rng.integers(1, 4, n).astype(float)  # few players -> repeated yellows
            frame.loc[rng.random(n) < 0.05, "player_id"] = np.nan
        null = rng.random(n) < 0.04
        if not null_actor_events:
            null &= kinds == 0
        frame["team_id"] = frame["team_id"].astype(object)
        frame.loc[null, "team_id"] = None if team_kind == "str" else np.nan
        frames.append(frame)
    actions = pd.concat(frames, ignore_index=True)
    if team_kind == "int":
        actions["team_id"] = pd.to_numeric(actions["team_id"])  # float64 with NaN
    elif team_kind == "nullable":
        actions["team_id"] = pd.array(pd.to_numeric(actions["team_id"]), dtype="Int64")
    if object_column:
        actions["original_event_id"] = [f"e{i}" for i in range(len(actions))]
    return actions


def _games_for(actions: pd.DataFrame) -> pd.DataFrame:
    """Home = the first listed team of each game; game 1 has no row (its home team is unknown)."""
    rows = []
    for gid, g in actions.groupby("game_id", sort=False):
        teams = list(pd.unique(g["team_id"].dropna()))
        if teams and gid not in (1, "game1"):
            rows.append((gid, sorted(teams, key=str)[0]))
    return pd.DataFrame(rows, columns=["game_id", "home_team_id"])


def _assert_same_states(actions: pd.DataFrame, games: pd.DataFrame | None) -> pd.DataFrame:
    try:
        want = _derive_oracle(actions, games=games, params=_P)
    except Exception as exc:
        with pytest.raises(type(exc)) as info:
            derive_match_states(actions, games=games, params=_P)
        assert str(info.value) == str(exc)
        return pd.DataFrame()
    got = derive_match_states(actions, games=games, params=_P)
    pd.testing.assert_frame_equal(got, want, check_exact=True)
    assert [type(v) for v in got["team_id"]] == [type(v) for v in want["team_id"]]
    assert [type(v) for v in got["game_id"]] == [type(v) for v in want["game_id"]]
    return want


_CASES = [
    {"team_kind": "int", "object_column": False},
    {"team_kind": "int", "object_column": True},
    {"team_kind": "str", "object_column": True},
    {"team_kind": "nullable", "object_column": True},
    {"team_kind": "nullable", "object_column": False},
    {"team_kind": "int", "object_column": True, "player_column": False},
    {"team_kind": "int", "object_column": False, "null_actor_events": True},
    {"team_kind": "str", "object_column": True, "null_actor_events": True},
    {"team_kind": "nullable", "object_column": True, "null_actor_events": True},
]


@pytest.mark.parametrize("case", range(len(_CASES)))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_derive_match_states_matches_the_oracle(case, seed):
    actions = _random_actions(seed, **_CASES[case])
    _assert_same_states(actions, _games_for(actions))


def test_home_team_from_an_actions_column_and_a_missing_games_frame():
    actions = _random_actions(5, team_kind="str")
    home = _games_for(actions).set_index("game_id")["home_team_id"]
    _assert_same_states(actions.assign(home_team_id=actions["game_id"].map(home)), None)
    _assert_same_states(actions, None)  # neither -> the same ValueError


def test_degenerate_inputs_behave_like_the_oracle():
    actions = _random_actions(6, team_kind="int")
    _assert_same_states(actions.iloc[:0], _games_for(actions))  # empty -> the same KeyError
    one_row = actions.iloc[[0]]
    _assert_same_states(one_row, _games_for(actions))
    no_team = actions.assign(team_id=np.nan)
    _assert_same_states(no_team, _games_for(actions))
    nan_game = actions.copy()
    nan_game["game_id"] = nan_game["game_id"].astype(object)
    nan_game.loc[nan_game.index[:30], "game_id"] = None  # dropped by the groupby
    _assert_same_states(nan_game, _games_for(actions))


def test_mixed_int_and_str_team_ids_follow_same_id():
    actions = _random_actions(7, team_kind="int", object_column=True)
    mixed = actions.astype({"team_id": object})
    mixed.loc[mixed.index[::3], "team_id"] = mixed.loc[mixed.index[::3], "team_id"].map(
        lambda v: v if pd.isna(v) else str(int(v))
    )
    _assert_same_states(mixed, _games_for(actions))


def test_derive_match_states_runs_no_iterrows(monkeypatch):
    def refuse(self):
        raise AssertionError("derive_match_states walked rows with iterrows")

    monkeypatch.setattr(pd.DataFrame, "iterrows", refuse)
    actions = _random_actions(8, team_kind="int")
    derive_match_states(actions, games=_games_for(actions), params=_P)


# --------------------------------------------------------------------------------------------- scoring


def _model_and_strength(actions: pd.DataFrame, games: pd.DataFrame, seed: int):
    rng = np.random.default_rng(seed)
    two = actions[
        actions["game_id"].isin([g for g, n in actions.groupby("game_id")["team_id"].nunique().items() if n == 2])
    ]
    model = WinProbabilityModel(params=_P).fit(two, games=games)
    xs = np.sort(rng.uniform(0.0, 1.0, 12))
    model._isotonic = (xs, np.sort(rng.uniform(0.0, 1.0, 12)))  # a monotone recalibration
    keys = actions[["game_id", "team_id"]].drop_duplicates()
    with_strength = actions.merge(
        keys.assign(base_strength=rng.normal(0.0, 0.3, len(keys))), on=["game_id", "team_id"], how="left"
    )
    return model, _strength_map(with_strength, "base_strength"), with_strength


@pytest.mark.parametrize("case", [0, 1, 2, 3])  # case 6 raises in derive (a null-actor yellow card), as the oracle does
@pytest.mark.parametrize("seed", [0, 3])
def test_score_states_matches_the_oracle_byte_for_byte(case, seed):
    actions = _random_actions(seed, **_CASES[case])
    games = _games_for(actions)
    states = derive_match_states(actions, games=games, params=_P).reset_index(drop=True)
    states["_pos"] = np.arange(len(states))
    model, strength, _ = _model_and_strength(actions, games, seed)
    want = _score_oracle(states, model, strength, _P)
    got = _score_states(states, model, strength, _P)
    for g_arr, w_arr in zip(got, want, strict=True):
        assert g_arr.dtype == w_arr.dtype
        assert g_arr.tobytes() == w_arr.tobytes()


def test_score_states_without_isotonic_matches_the_oracle():
    actions = _random_actions(9, team_kind="str")
    games = _games_for(actions)
    states = derive_match_states(actions, games=games, params=_P).reset_index(drop=True)
    states["_pos"] = np.arange(len(states))
    model, strength, _ = _model_and_strength(actions, games, 9)
    model._isotonic = None
    for g_arr, w_arr in zip(
        _score_states(states, model, strength, _P), _score_oracle(states, model, strength, _P), strict=True
    ):
        assert g_arr.tobytes() == w_arr.tobytes()


def test_score_states_runs_no_iterrows(monkeypatch):
    actions = _random_actions(10, team_kind="int")
    games = _games_for(actions)
    states = derive_match_states(actions, games=games, params=_P).reset_index(drop=True)
    states["_pos"] = np.arange(len(states))
    model, strength, _ = _model_and_strength(actions, games, 10)

    def refuse(self):
        raise AssertionError("_score_states walked rows with iterrows")

    monkeypatch.setattr(pd.DataFrame, "iterrows", refuse)
    _score_states(states, model, strength, _P)


# --------------------------------------------------------------------------------------------- the public calls


@pytest.mark.parametrize("case", [0, 2, 3, 7])
def test_compute_win_probability_and_goal_leverage_match_the_oracle_path(case, monkeypatch):
    actions = _random_actions(11, **_CASES[case])
    games = _games_for(actions)
    model, _, with_strength = _model_and_strength(actions, games, 11)
    got, got_report = compute_win_probability(with_strength, model=model, games=games, strength_column="base_strength")
    got_lev = goal_leverage(with_strength, model=model, games=games, strength_column="base_strength")
    monkeypatch.setattr(compute_mod, "derive_match_states", _derive_oracle)
    monkeypatch.setattr(compute_mod, "_score_states", _score_oracle)
    want, want_report = compute_win_probability(
        with_strength, model=model, games=games, strength_column="base_strength"
    )
    want_lev = goal_leverage(with_strength, model=model, games=games, strength_column="base_strength")
    pd.testing.assert_frame_equal(got, want, check_exact=True)
    assert got_report == want_report
    pd.testing.assert_series_equal(got_lev, want_lev, check_exact=True)


def test_both_derive_paths_are_exercised(monkeypatch):
    """Non-vacuity: the identity cases above run the array path AND the exact walk (null-actor events)."""
    import silly_kicks.win_probability._state as state_mod

    calls = {"walk": 0}
    real_walk = state_mod._game_states_walk

    def counting(game):
        calls["walk"] += 1
        return real_walk(game)

    monkeypatch.setattr(state_mod, "_game_states_walk", counting)
    plain = _random_actions(0, **_CASES[1])
    _assert_same_states(plain, _games_for(plain))
    assert calls["walk"] == 0, "a plain corpus must take the array path in every game"
    events = _random_actions(0, team_kind="str", object_column=True, null_actor_events=True, player_column=False)
    _assert_same_states(events, _games_for(events))
    assert calls["walk"] > 0, "a null-actor goal or red card must take the exact walk"
