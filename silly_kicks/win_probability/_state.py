"""Event-only game-state derivation for the in-game win-probability model.

Per action, the PRE-action state: cumulative score (id-based, ADR-018 own-goal-by-result), absolute
match minute (rebuilt from the period-relative ``time_seconds``), ``score_diff`` in the acting-team
perspective, ``home`` flag, and the ``man_advantage`` red-card differential (direct red + second
yellow). No ``add_names``, no ``vaep`` import -- the id-based goal rule mirrors ``match_outcome``.
"""

from __future__ import annotations

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
