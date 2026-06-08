"""TF-27: synthetic, CI-runnable guards for the SkillCorner GK-roster verification harness.

No network. Exercises the SAME pure functions the e2e gate uses
(tests/_skillcorner_sample.py), so the comparator is CI-covered (the e2e itself is
-m "not e2e" and does not run in CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests._skillcorner_sample import (
    AgreementResult,
    build_skillcorner_gk_truth,
    compare_gk_picks,
)


def _meta(players):
    return {"players": players}


def _p(pid, team, acronym, short_name="X. Y"):
    return {"id": pid, "team_id": team, "short_name": short_name, "player_role": {"acronym": acronym}}


# --- build_skillcorner_gk_truth ---


def test_truth_one_gk_per_team_str_keyed():
    meta = _meta([_p(10, 100, "GK"), _p(11, 100, "CB"), _p(20, 200, "GK"), _p(21, 200, "SUB")])
    assert build_skillcorner_gk_truth(meta) == {"100": ["10"], "200": ["20"]}


def test_truth_two_rostered_gks_returns_both_no_raise():
    meta = _meta([_p(10, 100, "GK"), _p(12, 100, "GK")])
    assert build_skillcorner_gk_truth(meta) == {"100": ["10", "12"]}


def test_truth_zero_gk_team_omitted():
    meta = _meta([_p(11, 100, "CB"), _p(20, 200, "GK")])
    truth = build_skillcorner_gk_truth(meta)
    assert "100" not in truth and truth["200"] == ["20"]


# --- compare_gk_picks ---


def test_compare_exact_match_is_perfect():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10"], (999, 200): ["20"]}  # int team key, int->str cast required
    r = compare_gk_picks(truth, picks, match_id=999)
    assert r.is_perfect and len(r.matched) == 2 and not r.mismatched


def test_compare_over_identification_fails_by_default():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10", "11"], (999, 200): ["20"]}  # starter + outfielder
    r = compare_gk_picks(truth, picks, match_id=999)
    assert not r.is_perfect and len(r.mismatched) == 1
    assert r.mismatched[0].team_id == "100"


def test_compare_allowlisted_over_identification_passes_via_subset():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10", "11"], (999, 200): ["20"]}
    r = compare_gk_picks(truth, picks, match_id=999, subset_allowlist=frozenset({("999", "100")}))
    assert r.is_perfect


def test_compare_wrong_pick_fails():
    truth = {"100": ["10"]}
    picks = {(999, 100): ["11"]}
    assert not compare_gk_picks(truth, picks, match_id=999).is_perfect


def test_compare_no_roster_gk_reported_not_failed():
    truth = {"100": ["10"]}  # team 200 has no rostered GK
    picks = {(999, 100): ["10"], (999, 200): ["20"]}
    r = compare_gk_picks(truth, picks, match_id=999)
    assert r.is_perfect and ("999", "200") in r.no_roster_gk


def test_compare_truth_team_missing_from_derived_fails():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10"]}  # derive found no GK for 200
    r = compare_gk_picks(truth, picks, match_id=999)
    assert not r.is_perfect and any(m.team_id == "200" for m in r.mismatched)


def test_cross_match_same_team_id_no_contamination():
    # Two matches share team 100 with DIFFERENT GKs. Per-match compare must pass;
    # a merged-truth implementation (last-match-win) would cross-validate and fail.
    rA = compare_gk_picks({"100": ["10"]}, {(1, 100): ["10"]}, match_id=1)
    rB = compare_gk_picks({"100": ["99"]}, {(2, 100): ["99"]}, match_id=2)
    agg = rA + rB
    assert agg.is_perfect and len(agg.matched) == 2


def test_agreement_result_empty_identity():
    e = AgreementResult.empty()
    assert e.is_perfect and (e + e).is_perfect


# --- synthetic derive_goalkeepers -> compare_gk_picks (CI wiring, no network) ---

from silly_kicks.tracking._gk_identification import derive_goalkeepers  # noqa: E402


def _planted_frames(n_frames: int = 12) -> pd.DataFrame:
    """Two teams; one planted GK each dwelling in its penalty area near goal.

    GK criterion (derive_goalkeepers): in PA (x<16.5 or x>88.5, 13.84<=y<=54.16) for
    >=40% of frames AND mean dist-to-nearest-goal-line <20m. Team-1 GK at x~3, team-2
    GK at x~102; outfielders at midfield (x~52) so they fail both criteria.
    """
    rows = []
    for f in range(1, n_frames + 1):
        base = dict(game_id="g1", period_id=1, frame_id=f, time_seconds=float(f), frame_rate=10.0, is_ball=False, z=0.0)
        # team 1: GK pid=10 near x=3; two outfielders at midfield
        rows.append({**base, "team_id": "100", "player_id": "10", "x": 3.0, "y": 34.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "100", "player_id": "11", "x": 52.0, "y": 30.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "100", "player_id": "12", "x": 60.0, "y": 40.0, "is_goalkeeper": False})
        # team 2: GK pid=20 near x=102; two outfielders at midfield
        rows.append({**base, "team_id": "200", "player_id": "20", "x": 102.0, "y": 34.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "200", "player_id": "21", "x": 53.0, "y": 38.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "200", "player_id": "22", "x": 45.0, "y": 30.0, "is_goalkeeper": False})
        # ball
        rows.append(
            {
                **base,
                "team_id": np.nan,
                "player_id": np.nan,
                "x": 50.0,
                "y": 34.0,
                "is_ball": True,
                "is_goalkeeper": False,
            }
        )
    return pd.DataFrame(rows)


def test_derive_then_compare_perfect_on_planted_frames():
    frames = _planted_frames()
    _out, picks = derive_goalkeepers(frames)
    truth = {"100": ["10"], "200": ["20"]}
    result = compare_gk_picks(truth, picks, match_id="g1")
    assert result.is_perfect, result.summary()
