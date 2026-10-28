"""Event-only game-state derivation for the in-game win-probability model.

Per action, the PRE-action state: cumulative score (id-based, ADR-018 own-goal-by-result), absolute
match minute (rebuilt from the period-relative ``time_seconds``), ``score_diff`` in the acting-team
perspective, ``home`` flag, and the ``man_advantage`` red-card differential (direct red + second
yellow). No ``add_names``, no ``vaep`` import -- the id-based goal rule mirrors ``match_outcome``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.spadl import config as _spadl

from ._config import WinProbabilityParams

_SHOT_TYPE_IDS = [_spadl.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_SUCCESS = _spadl.result_id["success"]
_OWNGOAL = _spadl.result_id["owngoal"]
_RED = _spadl.result_id["red_card"]
_YELLOW = _spadl.result_id["yellow_card"]
_FOUL = _spadl.actiontype_id["foul"]

# Absolute-minute offset per period start (regulation + extra time). Period 5 (PSO) excluded upstream.
_PERIOD_OFFSET_MIN = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}

#: The output columns, in the order the per-row body wrote them.
_STATE_COLUMNS = (
    "game_id",
    "action_id",
    "period_id",
    "team_id",
    "absolute_minute",
    "minutes_remaining",
    "own_score",
    "opp_score",
    "score_diff",
    "home",
    "man_advantage",
    "state_source",
)


def _home_team_map(actions: pd.DataFrame, games: pd.DataFrame | None) -> dict:
    if games is not None:
        return dict(zip(games["game_id"].astype(object), games["home_team_id"], strict=False))
    if "home_team_id" in actions.columns:
        return {g: h for g, h in zip(actions["game_id"], actions["home_team_id"], strict=False)}
    raise ValueError(
        "derive_match_states: home_team_id required via `games` (game_id, home_team_id) "
        "or an actions['home_team_id'] column"
    )


def derive_match_states(
    actions: pd.DataFrame,
    *,
    games: pd.DataFrame | None = None,
    params: WinProbabilityParams,
) -> pd.DataFrame:
    """Per-action PRE-action game state. Pure -- never mutates ``actions``.

    Computed per game over arrays (HFU plan R3-8 S3 item 7): the running score and red-card tallies are prefix
    sums of the event rows, the opponent and the home flag are resolved once per distinct team value, and the frame
    is built from the same Python values a row walk wrote. Each row reads its values from the game's ``.values``,
    the array ``iterrows`` reads, so every value keeps its type. A game in which a row without a team scores, is
    sent off or is booked with a player falls back to an exact walk over those arrays, because there the running
    tallies key on the missing value itself.
    """
    home_map = _home_team_map(actions, games)
    a = actions.sort_values(["game_id", "period_id", "time_seconds", "action_id"]).reset_index(drop=True)

    columns: list[list] = [[] for _ in _STATE_COLUMNS]
    for game_id, g in a.groupby("game_id", sort=False):
        part = _game_states(game_id, g, home_map.get(game_id), params)
        for out, values in zip(columns, part, strict=True):
            out.extend(values)
    if not columns[0]:
        return pd.DataFrame([]).astype({"man_advantage": "int64", "home": "boolean"})
    rows = list(zip(*columns, strict=True))
    return pd.DataFrame(rows, columns=list(_STATE_COLUMNS)).astype({"man_advantage": "int64", "home": "boolean"})


def _final_minute(g: pd.DataFrame, params: WinProbabilityParams) -> float:
    period_np = g["period_id"].to_numpy()
    time_np = g["time_seconds"].to_numpy()
    last_period = int(period_np.max())
    last_time = float(time_np[period_np == last_period].max())
    final_min = _PERIOD_OFFSET_MIN.get(last_period, 90) + last_time / 60.0
    return max(final_min, params.regulation_minutes)


class _Game:
    """One game's inputs, read once: the walk's dicts key on these values."""

    def __init__(self, game_id, g: pd.DataFrame, home_id, params: WinProbabilityParams) -> None:
        self.game_id = game_id
        self.home_id = home_id
        self.n = len(g)
        self.teams = list(pd.unique(g["team_id"].dropna()))
        self.scored = (g["type_id"].isin(_SHOT_TYPE_IDS) & (g["result_id"] == _SUCCESS)).to_numpy()
        self.own_goal = (g["result_id"] == _OWNGOAL).to_numpy()
        self.final_min = _final_minute(g, params)
        vals = g.values  # the array iterrows reads: each value keeps the type a row walk sees
        names = list(g.columns)
        self.team = vals[:, names.index("team_id")]
        self.type = vals[:, names.index("type_id")]
        self.result = vals[:, names.index("result_id")]
        self.period = vals[:, names.index("period_id")]
        self.time = vals[:, names.index("time_seconds")]
        self.action = vals[:, names.index("action_id")]
        self.player = vals[:, names.index("player_id")] if "player_id" in names else None


def _game_states(game_id, g: pd.DataFrame, home_id, params: WinProbabilityParams) -> tuple[list, ...]:
    """One game's state columns: the array path, or the exact walk where a tally keys on a missing team."""
    game = _Game(game_id, g, home_id, params)
    fast = _game_states_arrays(game)
    return fast if fast is not None else _game_states_walk(game)


def _opponent(teams: list, t):
    opp_candidates = [x for x in teams if not same_id(x, t)]
    return opp_candidates[0] if len(opp_candidates) == 1 else None


def _game_states_arrays(game: _Game) -> tuple[list, ...] | None:
    """The array path, or None when a tally would key on a missing team (the walk then serves the game)."""
    n, teams = game.n, game.teams
    lookup = {t: i for i, t in enumerate(teams)}
    if len(lookup) != len(teams):  # two team values, one dict key: the walk's dicts merge them
        return None
    if pd.isna(game.type).any() or pd.isna(game.result).any():
        return None
    n_teams = len(teams)
    ti = np.fromiter((lookup.get(t, -1) for t in game.team), dtype=np.int64, count=n)
    missing = ti < 0
    if not all(pd.isna(t) for t in game.team[missing]):
        return None

    # The opponent and the home flag depend on a row only through its team value: resolve them once per
    # (team, value type) class, with the walk's own expressions on that class's first value.
    type_code: dict = {}
    kinds = np.fromiter((type_code.setdefault(type(t), len(type_code)) for t in game.team), dtype=np.int64, count=n)
    classes = (ti + 1) * max(len(type_code), 1) + kinds
    _, first, inverse = np.unique(classes, return_index=True, return_inverse=True)
    opp_of_class = []
    home_of_class = []
    for row in first:
        t = game.team[row]
        opp = _opponent(teams, t)
        opp_of_class.append(n_teams if opp is None else lookup[opp])
        home_of_class.append(game.home_id is not None and same_id(t, game.home_id))
    opp_idx = np.asarray(opp_of_class, dtype=np.int64)[inverse]
    has_opp = opp_idx < n_teams
    team_idx = np.where(missing, n_teams, ti)

    # Running tallies: increments at the event rows (a few per game), then exclusive prefix sums. Row n_teams of
    # each table stays zero: a missing team, or no opponent.
    is_foul = game.type == _FOUL
    red = is_foul & (game.result == _RED)
    yellow = is_foul & ~red & (game.result == _YELLOW)
    goals_inc = np.zeros((n_teams + 1, n), dtype=np.int64)
    reds_inc = np.zeros((n_teams + 1, n), dtype=np.int64)
    books: list[dict] = [{} for _ in range(n_teams)]
    for j in np.flatnonzero(game.scored | game.own_goal | red | yellow):
        i = int(ti[j])
        if bool(game.scored[j]):
            if i < 0:
                return None
            goals_inc[i, j] += 1
        elif bool(game.own_goal[j]) and has_opp[j]:
            goals_inc[opp_idx[j], j] += 1
        if red[j]:
            if i < 0:
                return None
            reds_inc[i, j] += 1
        elif yellow[j]:
            pid = game.player[j] if game.player is not None else None
            if pid is not None and not pd.isna(pid):
                if i < 0:
                    return None
                book = books[i]
                book[pid] = book.get(pid, 0) + 1
                if book[pid] >= 2:
                    reds_inc[i, j] += 1  # second yellow -> red
    goals_before = np.cumsum(goals_inc, axis=1) - goals_inc
    reds_before = np.cumsum(reds_inc, axis=1) - reds_inc
    cols = np.arange(n)
    own_s = goals_before[team_idx, cols]
    opp_s = np.where(has_opp, goals_before[opp_idx, cols], 0)
    man = np.where(has_opp, reds_before[opp_idx, cols] - reds_before[team_idx, cols], 0)

    period_int = [int(p) for p in game.period]
    offset = np.fromiter((_PERIOD_OFFSET_MIN.get(p, 0) for p in period_int), dtype=np.int64, count=n)
    abs_min = offset + np.asarray([float(x) for x in game.time], dtype=np.float64) / 60.0
    remaining = game.final_min - abs_min
    remaining = np.where(0.0 > remaining, 0.0, remaining)  # max(x, 0.0), NaN and -0.0 included
    resolved = (game.home_id is not None) & has_opp
    return (
        [game.game_id] * n,
        [int(x) for x in game.action],
        period_int,
        list(game.team),
        abs_min.tolist(),
        remaining.tolist(),
        own_s.tolist(),
        opp_s.tolist(),
        (own_s - opp_s).tolist(),
        [home_of_class[k] for k in inverse],
        man.tolist(),
        np.where(resolved, "resolved", "unresolved").tolist(),
    )


def _game_states_walk(game: _Game) -> tuple[list, ...]:
    """The exact row walk over the game's arrays: the `2474b73` per-row body, reading each value where `iterrows`
    read it (the game's ``.values``), without building a Series per row."""
    teams, home_id, final_min = game.teams, game.home_id, game.final_min
    goals = {t: 0 for t in teams}
    reds = {t: 0 for t in teams}
    yellows: dict = {t: {} for t in teams}
    out: tuple[list, ...] = tuple([] for _ in _STATE_COLUMNS)
    for j in range(game.n):
        t = game.team[j]
        opp = _opponent(teams, t)
        abs_min = _PERIOD_OFFSET_MIN.get(int(game.period[j]), 0) + float(game.time[j]) / 60.0
        own_s = goals.get(t, 0)
        opp_s = goals.get(opp, 0) if opp is not None else 0
        man = (reds.get(opp, 0) - reds.get(t, 0)) if opp is not None else 0
        row = (
            game.game_id,
            int(game.action[j]),
            int(game.period[j]),
            t,
            abs_min,
            max(final_min - abs_min, 0.0),
            own_s,
            opp_s,
            own_s - opp_s,
            (home_id is not None and same_id(t, home_id)),
            int(man),
            "resolved" if (home_id is not None and opp is not None) else "unresolved",
        )
        for column, value in zip(out, row, strict=True):
            column.append(value)
        # advance running tallies AFTER recording the pre-action state
        if bool(game.scored[j]):
            goals[t] = goals.get(t, 0) + 1
        elif bool(game.own_goal[j]) and opp is not None:
            goals[opp] = goals.get(opp, 0) + 1
        if game.type[j] == _FOUL and game.result[j] == _RED:
            reds[t] = reds.get(t, 0) + 1
        elif game.type[j] == _FOUL and game.result[j] == _YELLOW:
            pid = game.player[j] if game.player is not None else None
            if pid is not None and not pd.isna(pid):
                yellows[t][pid] = yellows[t].get(pid, 0) + 1
                if yellows[t][pid] >= 2:
                    reds[t] = reds.get(t, 0) + 1  # second yellow -> red
    return out
