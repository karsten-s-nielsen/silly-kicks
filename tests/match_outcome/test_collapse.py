"""Rung-3b possession collapse (TF-53 Task 3)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.match_outcome import MatchOutcomeParams, compute_match_outcome
from silly_kicks.match_outcome._collapse import collapse_team_xgs
from tests.match_outcome._helpers import PASS, SHOT, make_actions, rows_by

# Isolate the collapse axis: both hold team_dependence="independent" so only same_possession differs
# (the package DEFAULT now sets team_dependence="dixon_coles" too, ADR-097).
_INDEP = MatchOutcomeParams(same_possession="independent", team_dependence="independent")
_COLLAPSE_ONLY = MatchOutcomeParams(same_possession="collapse", team_dependence="independent")


def test_collapse_combines_same_possession():
    shots = pd.DataFrame({"xg": [0.3, 0.4, 0.2], "possession_id": [1, 1, 2]})
    out = np.sort(collapse_team_xgs(shots, xg_column="xg"))
    # possession 1: 1-(0.7*0.6)=0.58 ; possession 2: 0.2
    np.testing.assert_allclose(out, np.sort(np.array([0.58, 0.2])))


def test_collapse_noop_when_each_shot_own_possession():
    shots = pd.DataFrame({"xg": [0.3, 0.4, 0.2], "possession_id": [1, 2, 3]})
    np.testing.assert_allclose(np.sort(collapse_team_xgs(shots, xg_column="xg")), np.sort([0.3, 0.4, 0.2]))


def test_collapse_drops_nan_xg():
    shots = pd.DataFrame({"xg": [0.3, float("nan")], "possession_id": [1, 1]})
    np.testing.assert_allclose(collapse_team_xgs(shots, xg_column="xg"), np.array([0.3]))


def test_compute_collapse_differs_on_same_possession():
    # team 10: two shots 0.5s apart, same possession (no opponent between, gap < 7s)
    fx = make_actions(
        [
            {"team_id": 10, "type_id": SHOT, "xg": 0.5, "time_seconds": 0.0},
            {"team_id": 10, "type_id": SHOT, "xg": 0.5, "time_seconds": 0.5},
            {"team_id": 20, "type_id": SHOT, "xg": 0.3, "time_seconds": 30.0},
        ]
    )
    indep = rows_by(compute_match_outcome(fx, xg_column="xg", params=_INDEP)[0])
    coll = rows_by(compute_match_outcome(fx, xg_column="xg", params=_COLLAPSE_ONLY)[0])
    # independent team10 pmf = PB([0.5,0.5]); collapsed = PB([0.75]) -> different p_win
    assert abs(indep[10]["p_win"] - coll[10]["p_win"]) > 1e-3
    for r in (indep, coll):
        r10 = r[10]
        assert abs(r10["p_win"] + r10["p_draw"] + r10["p_loss"] - 1.0) < 1e-12


def test_compute_collapse_identical_when_all_distinct():
    # opponent action between the two team-10 shots -> two distinct possessions -> collapse is a no-op
    fx = make_actions(
        [
            {"team_id": 10, "type_id": SHOT, "xg": 0.5, "time_seconds": 0.0},
            {"team_id": 20, "type_id": PASS, "time_seconds": 1.0},
            {"team_id": 10, "type_id": SHOT, "xg": 0.5, "time_seconds": 2.0},
            {"team_id": 20, "type_id": SHOT, "xg": 0.3, "time_seconds": 30.0},
        ]
    )
    indep = compute_match_outcome(fx, xg_column="xg", params=_INDEP)[0].sort_values("team_id").reset_index(drop=True)
    coll = (
        compute_match_outcome(fx, xg_column="xg", params=_COLLAPSE_ONLY)[0]
        .sort_values("team_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(indep, coll)
