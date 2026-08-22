"""Task 2: the load-bearing leakage guard for the receiver feature extractor.

Pinned as OUTPUT-INVARIANCE (not the disjunctive "raises or ignores", L1): perturbing the action's
end/loss location must leave the features BYTE-IDENTICAL, proving the extractor cannot read the
outcome-selected end. A reverse control proves the invariance is not vacuous.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.tracking._receiver import receiver_candidate_features

_ATT, _DEF = 1, 2


def _frame() -> pd.DataFrame:
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 10.0, 0.0),
        (False, 9, _ATT, False, 50.0, 34.0, 0.0, 0.0),
        (False, 10, _ATT, False, 70.0, 34.0, 2.0, 0.0),
        (False, 11, _ATT, False, 55.0, 50.0, 0.0, 0.0),
        (False, 20, _DEF, False, 60.0, 34.0, 0.0, 0.0),
        (False, 30, _DEF, True, 100.0, 34.0, 0.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
    df["game_id"], df["period_id"], df["frame_id"] = 1, 1, 100
    return df.astype({"player_id": "Int64", "team_id": "Int64"})


def _action() -> pd.Series:
    # carries an end location the extractor must NEVER read
    return pd.Series({"player_id": 9, "team_id": _ATT, "start_x": 50.0, "start_y": 34.0, "end_x": 71.0, "end_y": 34.0})


def test_features_invariant_to_end_location():
    frame = _frame()
    base = receiver_candidate_features(_action(), frame, feature_set="owner")
    moved_end = _action()
    moved_end["end_x"], moved_end["end_y"] = 999.0, -999.0  # arbitrary outcome perturbation
    perturbed = receiver_candidate_features(moved_end, frame, feature_set="owner")
    pd.testing.assert_frame_equal(base, perturbed)  # end location cannot have been read


def test_reverse_control_prepass_input_moves_features():
    """Non-vacuity: perturbing a PRE-PASS input (teammate 10's position) DOES change the features."""
    frame = _frame()
    base = receiver_candidate_features(_action(), frame, feature_set="owner")
    frame2 = frame.copy()
    frame2.loc[frame2["player_id"] == 10, "x"] = 90.0
    moved = receiver_candidate_features(_action(), frame2, feature_set="owner")
    assert not base.equals(moved)
