"""TF-54b orchestrator: compute_territorial_defense e2e shape/conservation + arms + ADR-019 dtype."""

import pandas as pd
import pytest

# territorial_defense was DEMOTED to experimental (ADR-090); the metric is retained on the private
# ``._compute`` / ``._columns`` path for the redesign.
from silly_kicks.territorial_defense._columns import TD_SAMPLE_COLUMNS, TD_SOURCE_VALUES
from silly_kicks.territorial_defense._compute import compute_territorial_defense

from ._fixtures import make_e2e_fixture, make_fitted_xt, make_per_action_ltr_fixture


def test_compute_shape_conservation_and_both_arms():
    # make_e2e_fixture is match-oriented (home-attacks-right) -> match_ltr is its convention; this tests
    # the match-oriented mode's shape + conservation. (per_action_ltr scoring is tested below.)
    actions, frames = make_e2e_fixture()
    samples, report = compute_territorial_defense(actions, frames, xt=make_fitted_xt(), frame_convention="match_ltr")

    # one row per defender (D, #102); exact column set
    assert list(samples.columns) == ["game_id", "player_id", *TD_SAMPLE_COLUMNS]
    assert len(samples) == 1
    row = samples.iloc[0]
    assert row["player_id"] == 102

    # Arm A: 3 scored interception frames, positive suppression
    assert int(row["a_frames_scored"]) == 3
    assert row["a_threat_suppressed"] > 0.0
    # Arm B: 1 qualifying opponent pass into D's hull, positive suppression, slippage measurable
    assert int(row["b_frames_scored"]) == 1
    assert row["b_threat_suppressed"] > 0.0
    # slippage = the NOT-D rate (attribution error, lower = tighter); the contesting defender IS D
    # here, so the attribution is perfect -> 0.0 (IMPL-02).
    assert row["b_attribution_slippage"] == 0.0

    assert row["td_source"] == "scored"
    assert set(samples["td_source"]) <= TD_SOURCE_VALUES

    # conservation over the Arm-A domain (3 interceptions, all scored)
    assert report.n_frames_in == 3
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    assert report.n_frames_scored == 3
    # Arm-B conservation over qualifying (defender, in-hull opponent-pass) pairs (ADR-042): the one
    # in-hull pass is scored, nothing dropped.
    assert report.arm_b_n_scored + sum(report.arm_b_drop_reasons.values()) == report.arm_b_n_in
    assert report.arm_b_n_in == 1 and report.arm_b_n_scored == 1


def test_unfitted_xt_is_refused():
    actions, frames = make_e2e_fixture()
    with pytest.raises((ValueError, TypeError, NotImplementedError)):
        compute_territorial_defense(actions, frames, xt=None)


def test_id_dtype_invariance_string_ids_match_numeric(ids_as_str=True):
    """ADR-019: stringified ids yield the SAME defender + the SAME suppression values as numeric."""
    actions, frames = make_e2e_fixture()
    xt = make_fitted_xt()
    num_samples, _ = compute_territorial_defense(actions, frames, xt=xt, frame_convention="match_ltr")

    a2 = actions.copy()
    f2 = frames.copy()
    for col in ("team_id", "player_id"):
        a2[col] = a2[col].astype("string")
        f2[col] = f2[col].astype("string")
    str_samples, _ = compute_territorial_defense(a2, f2, xt=xt, frame_convention="match_ltr")

    assert len(str_samples) == len(num_samples) == 1
    n, s = num_samples.iloc[0], str_samples.iloc[0]
    assert str(n["player_id"]) == str(s["player_id"])
    assert abs(float(n["a_threat_suppressed"]) - float(s["a_threat_suppressed"])) < 1e-9
    assert abs(float(n["b_threat_suppressed"]) - float(s["b_threat_suppressed"])) < 1e-9
    assert int(n["a_frames_scored"]) == int(s["a_frames_scored"])
    assert int(n["b_frames_scored"]) == int(s["b_frames_scored"])


def test_arm_b_slippage_is_honest_nan_when_contesting_defender_is_anonymous():
    """IMPL-01: on a realistic SB360 opponent-pass frame the ACTOR is the passer, so the contesting
    team-1 defender is a NON-actor with an anonymous (snapshot-numbered) id. Arm B still SCORES the
    threat suppression, but the attribution (is-D) is un-measurable -> ``b_attribution_slippage`` is
    honest-NaN, NEVER a fabricated 0.0 (ADR-027). The e2e fixture gives D the actor role on the pass
    frame (masking this), so this test moves ``is_actor`` onto the passer to reproduce real SB360."""
    actions, frames = make_e2e_fixture()
    f = frames.copy()
    on_pass_frame = f["frame_id"] == 3  # the opponent pass (action_id 3, team 2)
    # Realistic SB360: the passer (#201, team 2) is the ACTOR; every other row (incl. D #102) is anonymous.
    f["is_actor"] = (f["is_actor"] & ~on_pass_frame) | (on_pass_frame & (f["player_id"] == 201))
    samples, _ = compute_territorial_defense(actions, f, xt=make_fitted_xt(), frame_convention="match_ltr")
    row = samples[samples["player_id"] == 102].iloc[0]
    assert int(row["b_frames_scored"]) == 1  # the pass still scores for Arm B (threat is measurable)
    assert pd.isna(row["b_attribution_slippage"])  # ...but attribution is un-measurable -> honest-NaN


def test_per_action_ltr_scores_both_arms():
    """The SB360 DEFAULT (per_action_ltr) on genuinely per-action-LTR frames: both arms SCORE + conserve.
    This is the convention real SB360 uses (each freeze-frame in its acting team's LTR)."""
    actions, frames = make_per_action_ltr_fixture()
    samples, report = compute_territorial_defense(actions, frames, xt=make_fitted_xt())  # default per_action_ltr
    assert len(samples) == 1  # D (#102)
    row = samples.iloc[0]
    assert row["player_id"] == 102
    assert int(row["a_frames_scored"]) == 3 and row["a_threat_suppressed"] > 0.0  # Arm A non-vacuity
    assert int(row["b_frames_scored"]) >= 1 and row["b_threat_suppressed"] > 0.0  # Arm B non-vacuity
    # conservation (ADR-042) under the default convention
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    assert report.n_frames_scored == 3
    assert report.arm_b_n_scored + sum(report.arm_b_drop_reasons.values()) == report.arm_b_n_in


def test_per_action_frames_score_under_convention():
    """Two-sided regression (the guard that was missing, spec §8.3): on mixed-acting-team per-action
    frames the per-MATCH resolution (match_ltr) is bimodal -> 0 scored, while per_action_ltr resolves
    per frame -> scores. Fails on the bug (per-match), passes on the fix -- and proves the two modes are
    genuinely different, not the same object."""
    actions, frames = make_per_action_ltr_fixture()
    xt = make_fitted_xt()
    s_match, _ = compute_territorial_defense(actions, frames, xt=xt, frame_convention="match_ltr")
    s_pa, _ = compute_territorial_defense(actions, frames, xt=xt, frame_convention="per_action_ltr")
    a_match = pd.to_numeric(s_match["a_threat_suppressed"], errors="coerce")
    a_pa = pd.to_numeric(s_pa["a_threat_suppressed"], errors="coerce")
    assert a_match.notna().sum() == 0  # per-match: bimodal GK -> all unresolved_geometry (the bug)
    assert a_pa.notna().sum() >= 1  # per-frame convention: scores (the fix)


def test_match_ltr_uses_the_per_match_resolution():
    """R2 (byte-identity golden): match_ltr's factory genuinely closes over resolve_defended_goals -- its
    goal ends equal the per-match map, and the mode still scores the match-oriented fixture. Guards a
    future refactor that silently changes match_ltr semantics."""
    from silly_kicks.territorial_defense import _compute as C
    from silly_kicks.tracking import resolve_defended_goals

    actions, frames = make_e2e_fixture()
    xt = make_fitted_xt()
    got, _ = compute_territorial_defense(actions, frames, xt=xt, frame_convention="match_ltr")
    factory = C._make_goal_map_for(frames, "match_ltr")
    ref = resolve_defended_goals(frames)
    for team in (1, 2):
        assert factory(1, 1, team, 3 - team).attacked_goal(1, 1, team, allow_guess=True) == ref.attacked_goal(
            1, 1, team, allow_guess=True
        )
    assert len(got) == 1 and int(got.iloc[0]["a_frames_scored"]) == 3  # match-oriented mode still scores
