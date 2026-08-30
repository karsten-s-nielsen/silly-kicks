"""compute_rest_defense / summarize_rest_defense (TF-60, ADR-080)."""

import pandas as pd

from silly_kicks.restdefense import RD_LAYER1_COLUMNS, RestDefenseReport
from silly_kicks.restdefense._columns import RD_GEOMETRY_SOURCE, RD_SAMPLE_KEYS
from silly_kicks.restdefense._compute import compute_rest_defense, summarize_rest_defense
from silly_kicks.tracking import resolve_defended_goals
from tests.restdefense._fixtures import make_rest_defense_fixture


def test_scored_rows_are_fully_populated_and_pinned():
    actions, frames = make_rest_defense_fixture()
    samples, report = compute_rest_defense(actions, frames)
    resolved = samples[samples[RD_GEOMETRY_SOURCE] == "resolved"]
    assert len(resolved) == 3  # a0, a1, a2
    for c in RD_LAYER1_COLUMNS:
        assert resolved[c].notna().all(), f"{c} has NaN on a resolved row"
    a0 = resolved[resolved["action_id"] == 0].iloc[0]
    assert a0["rd_num_superiority"] == 4
    assert a0["rd_num_superiority_gk"] == 5
    assert a0["rd_zone_occupancy"] == 3
    assert a0["rd_line_height"] == 24.0
    assert a0["rd_gk_to_line_distance"] == -19.0
    assert a0["rd_shape_2_3_vs_3_2"] == "4-1"
    assert isinstance(report, RestDefenseReport)


def test_output_schema_and_dtypes():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)
    expected = [
        *RD_SAMPLE_KEYS,
        "possession_id",
        "is_possession_loss",
        *RD_LAYER1_COLUMNS,
        RD_GEOMETRY_SOURCE,
    ]
    assert list(samples.columns) == expected
    for c in ("rd_num_superiority", "rd_num_superiority_gk", "rd_zone_occupancy"):
        assert str(samples[c].dtype) == "Int64"
    assert str(samples["rd_line_height"].dtype) == "float64"
    assert samples["is_possession_loss"].dtype == bool


def test_conservation():
    actions, frames = make_rest_defense_fixture()
    _, report = compute_rest_defense(actions, frames)
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    assert report.n_frames_in == len(actions)
    assert report.n_frames_scored == 3
    assert report.drop_reasons.get("not_committed_forward") == 1


def test_orientation_symmetry_in_full_pipeline():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)
    a0 = samples[samples["action_id"] == 0].iloc[0]  # home
    a2 = samples[samples["action_id"] == 2].iloc[0]  # away (point-reflection of a0)
    for c in RD_LAYER1_COLUMNS:
        assert a0[c] == a2[c], f"{c}: home {a0[c]} != away {a2[c]}"


def test_unresolvable_goal_yields_unresolved_nan_row_not_crash():
    actions, frames = make_rest_defense_fixture()
    # a GoalMap that cannot resolve this game's ends (built from a different game_id)
    other = frames.copy()
    other["game_id"] = 999
    bad_gm = resolve_defended_goals(other)
    samples, report = compute_rest_defense(actions, frames, goal_map=bad_gm)
    assert (samples[RD_GEOMETRY_SOURCE] == "unresolved").all()
    for c in RD_LAYER1_COLUMNS:
        assert samples[c].isna().all(), f"{c} should be NaN when geometry is unresolved"
    # still conserves; the unresolved rows are counted as goal_end_unresolved drops
    assert report.n_frames_scored == 0
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in


def test_summarize_possession_and_match():
    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)
    per_poss = summarize_rest_defense(samples, by="possession")
    per_match = summarize_rest_defense(samples, by="match")
    # 3 scored samples fall in 2 possessions (P0 has a0+a1, P1 has a2)
    assert len(per_poss) == 2
    assert "n_samples" in per_poss.columns
    # one row per (team, game): team1 (a0,a1) and team2 (a2)
    assert len(per_match) == 2
    assert set(per_match["team_id"]) == {1, 2}
    # mean numerical superiority for team1 = mean(a0=4, a1=2) = 3.0
    t1 = per_match[per_match["team_id"] == 1].iloc[0]
    assert t1["rd_num_superiority"] == 3.0


def test_rd_width_is_rearguard_lateral_not_whole_team():
    """Option B (owner-ratified 2026-08-30): rd_width is the BACK-LINE lateral width
    (compute_defensive_line.lateral_width), NOT the whole-team width. The fixture's wide forwards make
    the two differ (28 vs 38), so this genuinely pins the rearguard source rather than coinciding."""
    from silly_kicks.tracking import compute_team_shape

    actions, frames = make_rest_defense_fixture()
    ts = compute_team_shape(frames, team_id=1)
    whole_team_width = ts[ts["frame_id"] == 100]["team_width"].iloc[0]
    samples, _ = compute_rest_defense(actions, frames)
    a0 = samples[samples["action_id"] == 0].iloc[0]
    assert a0["rd_width"] == 28.0  # back-4 lateral width
    assert whole_team_width == 38.0 and a0["rd_width"] != whole_team_width  # genuinely distinct


def test_guessed_goal_end_is_labeled_guessed_not_resolved():
    """IMPL-02: a GoalMap end resolvable only via allow_guess (a keeper-less team, outfield-mean
    fallback) is labelled 'guessed', not 'resolved' -- an honest distinction on FOV-cropped SB360."""
    actions, frames = make_rest_defense_fixture()
    no_gk = frames[~((frames["team_id"] == 1) & frames["is_goalkeeper"])].reset_index(drop=True)
    samples, _ = compute_rest_defense(actions, no_gk)
    t1 = samples[(samples["team_id"] == 1) & (samples["rd_geometry_source"] != "unresolved")]
    t2 = samples[samples["team_id"] == 2]
    assert len(t1) >= 1 and (t1["rd_geometry_source"] == "guessed").all()  # keeper-less -> guessed
    assert len(t2) >= 1 and (t2["rd_geometry_source"] == "resolved").all()  # keeps its keeper


def test_num_superiority_na_on_non_two_team_game():
    """IMPL-04 end-to-end: a frame set without exactly two teams -> rd_num_superiority NaN across all
    scored rows (never a silent A-count); the opponent-free metrics still compute."""
    actions, frames = make_rest_defense_fixture()
    extra = frames.iloc[[0]].copy()
    extra["team_id"] = pd.array([3], dtype="Int64")
    extra["player_id"] = pd.array([301], dtype="Int64")
    extra["x"] = 50.0
    extra["is_goalkeeper"] = False
    frames3 = pd.concat([frames, extra], ignore_index=True)
    samples, _ = compute_rest_defense(actions, frames3)
    resolved = samples[samples["rd_geometry_source"] == "resolved"]
    assert len(resolved) >= 1
    assert resolved["rd_num_superiority"].isna().all()
    assert resolved["rd_num_superiority_gk"].isna().all()
    assert resolved["rd_zone_occupancy"].notna().any()  # opponent-free -> still computed


def test_rd_geometry_source_values_in_vocabulary():
    from silly_kicks.restdefense._columns import RD_GEOMETRY_SOURCE_VALUES

    actions, frames = make_rest_defense_fixture()
    samples, _ = compute_rest_defense(actions, frames)
    assert set(samples["rd_geometry_source"].dropna().unique()) <= set(RD_GEOMETRY_SOURCE_VALUES)


def test_purity_does_not_mutate_inputs():
    actions, frames = make_rest_defense_fixture()
    a_before, f_before = actions.copy(), frames.copy()
    compute_rest_defense(actions, frames)
    pd.testing.assert_frame_equal(actions, a_before)
    pd.testing.assert_frame_equal(frames, f_before)
