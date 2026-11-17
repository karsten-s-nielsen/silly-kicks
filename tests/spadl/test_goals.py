"""One goal predicate for the library (R52, action-time repair spec §4.12)."""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl._goals import goal_mask, owngoal_mask

_T = spadlconfig.actiontype_id
_R = spadlconfig.result_id


def _actions(types: list[str], results: list[str], *, names: bool) -> pd.DataFrame:
    df = pd.DataFrame({"type_id": [_T[t] for t in types], "result_id": [_R[r] for r in results]})
    if names:
        df["type_name"] = types
    return df


def test_goal_mask_shot_types():
    a = _actions(
        ["shot", "shot_penalty", "shot_freekick", "shot", "pass"],
        ["success", "success", "success", "fail", "success"],
        names=False,
    )
    assert goal_mask(a).tolist() == [True, True, True, False, False]


def test_owngoal_any_type():
    a = _actions(["bad_touch", "shot", "pass"], ["owngoal", "owngoal", "success"], names=False)
    assert owngoal_mask(a).tolist() == [True, True, False]


def test_type_id_and_type_name_agree():
    types = ["shot", "shot_penalty", "shot_freekick", "pass", "bad_touch"]
    results = ["success", "success", "fail", "success", "owngoal"]
    by_id = _actions(types, results, names=False)
    by_name = _actions(types, results, names=True).drop(columns=["type_id"])
    assert goal_mask(by_id).tolist() == goal_mask(by_name).tolist() == [True, True, False, False, False]
    assert owngoal_mask(by_id).tolist() == owngoal_mask(by_name).tolist() == [False, False, False, False, True]


def test_masks_read_names_on_a_name_only_table():
    # a name-only table (type_name / result_name, no ids): the ghost-outfield chirality probe and several callers
    # build such tables; the former score lookup read them too (Chesterton)
    import pandas as pd

    from silly_kicks.spadl._goals import goal_mask, owngoal_mask

    a = pd.DataFrame({"type_name": ["shot_penalty", "bad_touch", "pass"],
                      "result_name": ["success", "owngoal", "success"]})  # fmt: skip
    assert goal_mask(a).tolist() == [True, False, False]
    assert owngoal_mask(a).tolist() == [False, True, False]
