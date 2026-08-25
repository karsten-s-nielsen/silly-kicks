"""Exact-equivalence oracle + structural guard for the vectorized possession labels (ADR-068).

The vectorized `_scores_possession` / `_concedes_possession` must be BYTE-IDENTICAL to the prior
nested-loop implementation, which is kept here verbatim as `_ref_*`. Fixtures cover the three cases
F6 requires: a two-team possession, the goal/owngoal-scores-itself pass, and multiple downstream
same-team goals of decreasing xG (proves reverse-cumulative MAX, not first)."""

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadl
from silly_kicks.vaep.labels import (
    _concedes_possession,
    _is_goal,
    _is_owngoal,
    _other_team_scalar,
    _same_team_scalar,
    _scores_possession,
)
from tests._perf_structural import call_counter

_SUCCESS = spadl.result_id["success"]
_OWN = spadl.result_id["owngoal"]
_FAIL = spadl.result_id["fail"]


def _row(poss, team, kind, xg=0.0, game: int | None = 1):
    tn, rid = {"goal": ("shot", _SUCCESS), "owngoal": ("bad_touch", _OWN)}.get(kind, ("pass", _FAIL))
    return dict(game_id=game, possession_id=poss, team_id=team, type_name=tn, result_id=rid, xg=xg)


def _rich_corpus():
    rows = [
        # poss 0: TWO teams in one possession + a downstream same-team goal
        _row(0, "A", "pass"),
        _row(0, "A", "pass"),
        _row(0, "B", "pass"),
        _row(0, "A", "pass"),
        _row(0, "A", "goal", 0.5),
        # poss 1: multiple downstream same-team goals, DECREASING xg -> first pass scores max (0.3)
        _row(1, "A", "pass"),
        _row(1, "A", "goal", 0.3),
        _row(1, "A", "goal", 0.1),
        # poss 2: opponent owngoal downstream (scores for A) + the owngoal concedes itself for B
        _row(2, "A", "pass"),
        _row(2, "B", "owngoal", 0.2),
        # poss 3: a lone goal (self-scores) and a lone owngoal (self-concedes)
        _row(3, "A", "goal", 0.4),
        _row(4, "B", "owngoal", 0.15),
        # poss 5: a NULL-team row (must not be promoted to opponent) followed by a goal
        _row(5, None, "pass"),
        _row(5, "A", "goal", 0.25),
        # NaN group key: groupby(dropna=True) drops these rows, so the old nested loop never applied
        # their self-event -- a NaN-possession goal/owngoal must stay 0.0/False, NOT self-score
        # (byte-identity guard for the global self-event init; ADR-068). Covers both the scores
        # self_event=goal and concedes self_event=owngoal paths, and (via game=None) the game_id key.
        _row(None, "A", "goal", 0.35),
        _row(None, "B", "owngoal", 0.45),
        _row(6, "A", "goal", 0.55, game=None),
    ]
    return pd.DataFrame(rows)


# --- reference: the pre-ADR-068 nested-loop implementation, verbatim ---
def _ref(actions, xg_column, *, col, same_is_goal):
    goal = _is_goal(actions)
    owngoal = _is_owngoal(actions)
    team_id = actions["team_id"]
    if xg_column is not None:
        xg = actions.get(xg_column, pd.Series(0.0, index=actions.index)).fillna(0.0)
    group_cols = ["game_id", "possession_id"] if "game_id" in actions.columns else ["possession_id"]
    result = pd.Series(0.0, index=actions.index) if xg_column is not None else pd.Series(False, index=actions.index)
    # scores: self = goal; concedes: self = owngoal (the pairwise same/other logic is below)
    self_ev = goal if same_is_goal else owngoal
    for _key, grp in actions.groupby(group_cols):
        idx = grp.index
        for i, pos in enumerate(idx):
            for j_pos in idx[i + 1 :]:
                if goal.loc[j_pos]:
                    ok = (
                        _same_team_scalar(team_id.loc[pos], team_id.loc[j_pos])
                        if same_is_goal
                        else _other_team_scalar(team_id.loc[pos], team_id.loc[j_pos])
                    )
                    if ok:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
                if owngoal.loc[j_pos]:
                    ok = (
                        _other_team_scalar(team_id.loc[pos], team_id.loc[j_pos])
                        if same_is_goal
                        else _same_team_scalar(team_id.loc[pos], team_id.loc[j_pos])
                    )
                    if ok:
                        if xg_column is not None:
                            result.loc[pos] = max(result.loc[pos], xg.loc[j_pos])
                        else:
                            result.loc[pos] = True
                            break
        for pos in idx:
            if self_ev.loc[pos]:
                if xg_column is not None:
                    result.loc[pos] = max(result.loc[pos], xg.loc[pos])
                else:
                    result.loc[pos] = True
    return pd.DataFrame(result, columns=[col])


@pytest.mark.parametrize("xg_column", [None, "xg"])
def test_scores_possession_matches_reference(xg_column):
    a = _rich_corpus()
    pd.testing.assert_frame_equal(_scores_possession(a, xg_column), _ref(a, xg_column, col="scores", same_is_goal=True))


@pytest.mark.parametrize("xg_column", [None, "xg"])
def test_concedes_possession_matches_reference(xg_column):
    a = _rich_corpus()
    pd.testing.assert_frame_equal(
        _concedes_possession(a, xg_column), _ref(a, xg_column, col="concedes", same_is_goal=False)
    )


def test_decreasing_xg_takes_max_not_first():
    # poss 1: pass then goal 0.3 then goal 0.1 -> the pass must score 0.3 (max), not 0.1 (last/first).
    a = _rich_corpus()
    scores = _scores_possession(a, "xg")["scores"].to_numpy()
    pass_pos = a.index[(a["possession_id"] == 1) & (a["type_name"] == "pass")][0]
    assert scores[a.index.get_loc(pass_pos)] == pytest.approx(0.3)


def _single_possession(k):
    rows = [_row(0, "A", "pass") for _ in range(k)] + [_row(0, "A", "goal", 0.5)]
    return pd.DataFrame(rows)


def test_loc_count_is_scale_independent(monkeypatch):
    # ADR-068/F2: the vectorized path must not scale .loc calls with possession length (was O(k^2)).
    import pandas.core.indexing as _idx

    def _count(k):
        calls = call_counter(monkeypatch, _idx._LocIndexer, "__getitem__")
        _scores_possession(_single_possession(k), "xg")
        return calls["n"]

    assert _count(12) == _count(4)  # bounded / constant in k, not quadratic
