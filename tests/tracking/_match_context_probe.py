"""The fixed probe of the score and phase inputs.

Each rule that ``match_context_revision`` names is exercised: the period key, the goal rule with a penalty, a
direct free kick, an own goal and a shoot-out row, the SkillCorner running score with its effect rule, a derived
row and its fallback, a penalty row, and the phase codes, the 10 s window and an ``"order_only"`` row
. Home is team 1.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id
HOME = 1


def _rows(rows: list[tuple]) -> pd.DataFrame:
    """(period, time, team, type, result[, time_source[, provenance]])."""
    out = []
    for i, r in enumerate(rows):
        period, t, team, ty, res = r[:5]
        d = {"game_id": "probe", "action_id": i, "period_id": period, "time_seconds": float(t), "team_id": team,
             "type_id": _T[ty], "result_id": _R[res]}  # fmt: skip
        if len(r) > 5:
            d["time_source"] = r[5]
        if len(r) > 6:
            d["action_provenance"] = r[6]
        out.append(d)
    return pd.DataFrame(out)


def goal_rows_table() -> pd.DataFrame:
    return _rows([
        (1, 100.0, 1, "shot", "success"),            # home goal, first half
        (1, 900.0, 2, "shot_penalty", "success"),     # away penalty goal
        (1, 2000.0, 1, "pass", "success"),
        (2, 50.0, 2, "bad_touch", "owngoal"),        # away own goal -> home goal
        (2, 600.0, 2, "shot_freekick", "success"),    # away direct free-kick goal
        (2, 700.0, 1, "shot", "fail"),
        (3, 30.0, 1, "shot", "success"),             # extra time
        (5, 10.0, 2, "shot_penalty", "success"),      # shoot-out: no goal of the score
    ])  # fmt: skip


SCORE_QUERIES = [(1, 50.0), (1, 100.0), (1, 899.0), (1, 950.0), (2, 10.0), (2, 60.0), (2, 650.0), (3, 20.0),
                 (3, 40.0), (4, 5.0), (5, 100.0)]  # fmt: skip


def running_score_table(*, decrease: bool = False) -> pd.DataFrame:
    """SkillCorner-shaped native rows with the provider's running score (team_score, opponent_team_score)."""
    rows = [
        (1, 10.0, 1, "native", 0, 0),
        (1, 20.0, 1, "native", 1, 0),  # the higher score first shows here
        (1, 25.0, 2, "derived", 9, 9),  # a derived row is never read
        (1, 90.0, 2, "native", 0, 1),  # the next native row: the effect is here
        (2, 30.0, 1, "native", 1, 0),
        (2, 80.0, 2, "native", 1, 1 if not decrease else 0),  # decrease variant: the away side drops 1 -> 0
        (5, 10.0, 1, "native", 2, 1),  # shoot-out: a period-5 row is not read; the score stays 1-1
        (5, 30.0, 2, "native", 1, 2),  # carries the effect of the row before it, if it were read
    ]
    out = []
    for i, (period, t, team, prov, ts, os_) in enumerate(rows):
        out.append({"game_id": "probe", "action_id": i, "period_id": period, "time_seconds": t, "team_id": team,
                    "type_id": _T["shot"] if (decrease and i == 1) else _T["pass"], "result_id": _R["success"],
                    "action_provenance": prov, "time_source": "tracking", "team_score": ts,
                    "opponent_team_score": os_})  # fmt: skip
    return pd.DataFrame(out)


# (4, 5): a period after the last row and before period 5 -- the end-of-period-4 score is a period-5 value only
RUNNING_QUERIES = [(1, 15.0), (1, 50.0), (1, 90.0), (2, 50.0), (2, 100.0), (5, 40.0), (4, 5.0)]


def phase_table() -> pd.DataFrame:
    return _rows([
        (1, 100.0, 1, "freekick_crossed", "success", "placeholder", "derived"),
        (1, 200.0, 2, "goalkick", "success", "kick", "derived"),
        (1, 205.0, 2, "corner_crossed", "fail", "order_only", "derived"),  # opens no window, ends none
        (1, 300.0, 1, "throw_in", "success", "placeholder", "derived"),     # not a set piece of the phase
        (1, 400.0, 1, "corner_short", "success", "kick", "derived"),
        (2, 50.0, 2, "freekick_short", "success", "event", "native"),
    ])  # fmt: skip


# the last two sit AT the time of a set-piece row (the kick frame): the phase starts at its own row
PHASE_QUERIES = [(1, 99.0), (1, 105.0), (1, 110.0), (1, 111.0), (1, 203.0), (1, 207.0), (1, 212.0), (1, 302.0),
                 (1, 405.0), (2, 55.0), (2, 105.0), (1, 55.0), (1, 100.0), (1, 200.0)]  # fmt: skip


def probe_outputs() -> dict[str, list]:
    """The probe's outputs with the current code: every query of each lookup."""
    from silly_kicks.tracking._ghost_gk import _build_phase_lookup, _build_score_lookup

    goal = _build_score_lookup(goal_rows_table(), home_team_id=HOME)
    running = _build_score_lookup(running_score_table(), home_team_id=HOME)
    fallback = _build_score_lookup(running_score_table(decrease=True), home_team_id=HOME)
    phase = _build_phase_lookup(phase_table())
    return {
        "score_goal_rows": [float(goal("probe", p, t)) for p, t in SCORE_QUERIES],
        "score_running": [float(running("probe", p, t)) for p, t in RUNNING_QUERIES],
        "score_fallback": [float(fallback("probe", p, t)) for p, t in RUNNING_QUERIES],
        "phase": [int(phase("probe", p, t)) for p, t in PHASE_QUERIES],
    }
