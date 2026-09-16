"""compute_match_outcome (TF-53 Task 2): mirror-consistency, census, honest-NaN, face validity."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.match_outcome import compute_match_outcome
from tests.match_outcome._helpers import PASS, SHOT, make_actions, rows_by


def _two_team(xgs_home, xgs_away, *, home=10, away=20, game_id=1):
    recs = [{"team_id": home, "type_id": SHOT, "xg": x} for x in xgs_home]
    recs += [{"team_id": away, "type_id": SHOT, "xg": x} for x in xgs_away]
    return make_actions(recs, game_id=game_id)


def test_two_team_rows_are_mirror_consistent():
    s, _ = compute_match_outcome(_two_team([0.5, 0.2], [0.3]), xg_column="xg")
    r = rows_by(s)
    r10, r20 = r[10], r[20]
    assert r10["p_win"] == pytest.approx(r20["p_loss"])
    assert r10["p_loss"] == pytest.approx(r20["p_win"])
    assert r10["p_draw"] == pytest.approx(r20["p_draw"])
    assert r10["xpoints"] == pytest.approx(3 * r10["p_win"] + r10["p_draw"])


def test_report_conserves():
    fx = pd.concat([_two_team([0.5], [0.4], game_id=1), _two_team([0.2], [0.1], game_id=2)], ignore_index=True)
    s, rep = compute_match_outcome(fx, xg_column="xg")
    assert rep.n_matches_scored + rep.n_matches_excluded_not_two_teams == rep.n_matches_in == 2
    assert rep.n_shots_with_xg + rep.n_shots_null_xg == rep.n_shots == 4
    assert len(s) == 4


def test_non_two_team_excluded_and_counted():
    one = make_actions([{"team_id": 10, "type_id": SHOT, "xg": 0.5}], game_id=9)  # single team
    two = _two_team([0.3], [0.3], game_id=1)
    _, rep = compute_match_outcome(pd.concat([one, two], ignore_index=True), xg_column="xg")
    assert rep.n_matches_in == 2 and rep.n_matches_scored == 1 and rep.n_matches_excluded_not_two_teams == 1


def test_no_shots_team_scores_zero():
    fx = make_actions([{"team_id": 10, "type_id": SHOT, "xg": 0.6}, {"team_id": 20, "type_id": PASS}], game_id=1)
    s, _ = compute_match_outcome(fx, xg_column="xg")
    r = rows_by(s)
    r10, r20 = r[10], r[20]
    # team 20 has no shot -> P(0)=1; team 10 wins iff it scores its one shot
    assert r10["p_win"] == pytest.approx(0.6)
    assert r20["p_loss"] == pytest.approx(0.6)
    assert r10["p_draw"] == pytest.approx(0.4)
    assert r20["expected_goals"] == 0.0


def test_nan_xg_shot_excluded_but_counted():
    fx = _two_team([0.5, float("nan")], [0.3], game_id=1)  # one NaN-xg shot for team 10
    s, rep = compute_match_outcome(fx, xg_column="xg")
    assert rep.n_shots == 3 and rep.n_shots_with_xg == 2 and rep.n_shots_null_xg == 1
    # the NaN shot is dropped from the pmf -> team 10 pmf is that of a single 0.5 shot
    r = rows_by(s)
    assert r[10]["p_win"] == pytest.approx(0.35)  # 0.5*(1-0.3)=0.35


def test_missing_xg_column_is_honest_all_draw():
    s, _ = compute_match_outcome(_two_team([0.5], [0.4]), xg_column="not_a_column")
    r = rows_by(s)
    assert r[10]["p_draw"] == pytest.approx(1.0)  # no xG -> 0-0 with certainty


def test_face_validity_ordering_and_magnitude():
    # West Ham (0.41 win in the course) has HIGHER total xG than Arsenal (0.33)
    arsenal = [0.3, 0.25, 0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.1, 0.1]  # ~1.57
    westham = [0.4, 0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.1, 0.1]  # ~1.72
    s, _ = compute_match_outcome(_two_team(arsenal, westham), xg_column="xg")
    r = rows_by(s)
    r10, r20 = r[10], r[20]
    assert r20["p_win"] > r10["p_win"]  # higher-xG team more likely to win
    for col in ("p_win", "p_draw", "p_loss"):
        assert 0.15 < r10[col] < 0.55  # rough magnitude band (no tune-to-target)


def test_order_insensitive_and_pure():
    fx = _two_team([0.5, 0.2], [0.3, 0.1])
    before = fx.copy(deep=True)
    s1, _ = compute_match_outcome(fx, xg_column="xg")
    s2, _ = compute_match_outcome(fx.iloc[::-1].reset_index(drop=True), xg_column="xg")
    pd.testing.assert_frame_equal(
        s1.sort_values("team_id").reset_index(drop=True), s2.sort_values("team_id").reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(fx, before)  # input unmutated
