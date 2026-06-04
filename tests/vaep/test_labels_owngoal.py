"""VAEP label own-goal counting (Component 3): own goals count by RESULT, not via a shot-type gate."""

from pathlib import Path

import pandas as pd

from silly_kicks.spadl import config as spadl
from silly_kicks.vaep import labels


def _actions(type_names, result_ids, team_ids=(1, 1)):
    return pd.DataFrame(
        {
            "game_id": [1] * len(type_names),
            "period_id": [1] * len(type_names),
            "team_id": list(team_ids),
            "type_name": list(type_names),
            "result_id": list(result_ids),
        }
    )


def test_owngoal_bad_touch_counts_as_concede_for_acting_team():
    # team 1 acts (pass), then a bad_touch+owngoal by team 1 -> team 1 concedes.
    a = _actions(["pass", "bad_touch"], [spadl.result_id["success"], spadl.result_id["owngoal"]])
    conceded = labels.concedes(a, nr_actions=2)
    assert bool(conceded["concedes"].iloc[0]) is True


def test_cross_fail_then_shot_success_credits_goal():
    # round-1 #7 adjacency: a failed cross immediately followed by a synthetic shot+success
    # (the cross-goal shape) must still credit the goal to the acting team.
    a = _actions(["freekick_crossed", "shot_freekick"], [spadl.result_id["fail"], spadl.result_id["success"]])
    scored = labels.scores(a, nr_actions=2)
    assert bool(scored["scores"].iloc[0]) is True


def test_no_shot_gated_owngoal_predicate_survives():
    # The bug being fixed IS a copy-pasted shot-gated owngoal predicate. Prove no copy survives:
    # zero lines combine str.contains("shot") with the owngoal result. Meta-test — catches a missed
    # site (e.g. the originally-overlooked 339-340 / 416-417) on any future refactor.
    src = Path(labels.__file__).read_text(encoding="utf-8")
    offenders = [ln.strip() for ln in src.splitlines() if 'str.contains("shot")' in ln and "owngoal" in ln]
    assert offenders == [], f"shot-gated owngoal predicate(s) still present: {offenders}"
