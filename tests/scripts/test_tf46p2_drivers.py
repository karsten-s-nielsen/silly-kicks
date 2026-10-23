"""TF-46 Arm 1 Part 2 pre-flight driver cores (spec sections 5, 16.2, 16.3).

These exercise the PURE cores in ``scripts/_tf46p2_common.py`` on synthetic corpora; the driver ``main()``
entry points run on the DGX in the C3 phase. One test per plan Task 2.13 Step-1 obligation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import _tf46p2_common as C
from silly_kicks.tracking._restart_validation import RATIFIED, DataProfile

# --- folds, anchor, corpus guard ---------------------------------------------------------------


def test_lomo_folds_deterministic_and_leave_one_out():
    f1 = C.lomo_folds(["m1", "m2", "m3"], seed=46)
    f2 = C.lomo_folds(["m3", "m1", "m2"], seed=46)  # order-independent
    assert f1 == f2
    assert len(f1) == 3
    for train, held in f1:
        assert held not in train
        assert sorted([*train, held]) == ["m1", "m2", "m3"]
    assert C.lomo_folds(["only"], seed=46) == []


def test_simulate_skillcorner_anchor_brackets_the_kick():
    actions = pd.DataFrame(
        {
            "action_id": [0, 1, 2],
            "period_id": [1, 1, 1],
            "time_seconds": [10.0, 20.0, 30.0],
            "type_name": ["pass", "corner_crossed", "pass"],
        }
    )
    out = C.simulate_skillcorner_anchor(actions)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["kick_bracket_lo"] == 10.0 and row["kick_bracket_hi"] == 30.0  # prev event .. next touch


def _direction_checks(agrees):
    # one row per (match, period): matches m0, m1, ... with periods 1 and 2
    return pd.DataFrame(
        {
            "match_id": [f"m{i // 2}" for i in range(len(agrees))],
            "period_id": [1 + i % 2 for i in range(len(agrees))],
            "agrees": agrees,
        }
    )


def test_gi_direction_corpus_guard_both_sides():
    C.assert_gi_direction_corpus(_direction_checks([True] * 200))  # no raise
    dirty = [False, True, False, True, False] + [True] * 195  # 3 of 200 = 1.5 % > 1 %, no match fully mismatched
    with pytest.raises(AssertionError, match="share"):
        C.assert_gi_direction_corpus(_direction_checks(dirty), max_share=0.01)
    one = [False] + [True] * 199  # 0.5 %, a single period: noise, no raise
    C.assert_gi_direction_corpus(_direction_checks(one), max_share=0.01)


def test_gi_direction_corpus_guard_both_periods_of_one_match():
    # Spec 4.3: BOTH periods of one match mismatched is a systematic flip bug even under the 1 % share.
    both = [False, False] + [True] * 398  # 0.5 % share, but m0 has periods 1 AND 2 mismatched
    with pytest.raises(AssertionError, match="BOTH periods"):
        C.assert_gi_direction_corpus(_direction_checks(both), max_share=0.01)


def test_gi_direction_corpus_guard_counts_unresolved_periods_both_sides():
    # R.2 2026-10-23: a lost flip record leaves periods UNRESOLVED (never compared); above the ceiling that must
    # raise, else the guard passes vacuously. Unresolved rows never count as agreements or mismatches.
    lost = _direction_checks([True] * 200).assign(resolved=[i >= 6 for i in range(200)])  # 3 % unresolved
    with pytest.raises(AssertionError, match="unresolved"):
        C.assert_gi_direction_corpus(lost, max_share=0.01)
    few = _direction_checks([True] * 200).assign(resolved=[i != 0 for i in range(200)])  # 0.5 %: no raise
    C.assert_gi_direction_corpus(few, max_share=0.01)
    # an unresolved row whose placeholder `agrees` is False is excluded, not counted as a mismatch
    masked = _direction_checks([False, False] + [True] * 398).assign(resolved=[i >= 2 for i in range(400)])
    C.assert_gi_direction_corpus(masked, max_share=0.01)


def test_part2_match_carries_the_staged_flip_keyed_by_match_and_period():
    # the staged profile json keys periods as strings; the match restores (match_id, int period) keys
    from scripts._tf46p2_score import part2_match_from

    meta = {"provider": "skillcorner", "ltr_flip_by_period": {"1": False, "2": True, "3": None}}
    m = part2_match_from("skillcorner", "7", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), meta)
    assert m.flips == {("7", 1): False, ("7", 2): True, ("7", 3): None}
    empty = pd.DataFrame()
    assert part2_match_from("idsse", "8", empty, empty, empty, empty, {}).flips == {}  # no record -> none recorded


# --- truth extraction (section 5.6) ------------------------------------------------------------


def test_dfl_first_touch_rules():
    ev = pd.DataFrame(
        {
            "restart_id": [1, 1, 2, 2, 3, 3, 4, 4, 5],
            "seq": [0, 1, 0, 1, 0, 1, 0, 1, 0],
            "event_type": [
                "Delete",
                "Play",
                "TacklingGame",
                "Play",
                "Foul",
                "Play",
                "BallClaiming",
                "Play",
                "BallClaiming",
            ],
            "player_id": ["x", "p1", "w", "p2", "f", "p3", "bc", "bc", "solo"],
            "team_id": ["A", "A", "B", "A", "A", "B", "A", "A", "B"],
            "timestamp_seconds": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 5.0, 5.05, 9.0],
            "tackle_winner": [pd.NA, pd.NA, "w", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
            "tackle_winner_team": [pd.NA, pd.NA, "B", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    out = C.dfl_first_touch_truth(ev).set_index("restart_id")
    assert out.loc[1, "touch_class"] == "touch" and out.loc[1, "player_id"] == "p1"  # Delete dropped
    assert out.loc[2, "touch_class"] == "duel" and out.loc[2, "player_id"] == "w"  # TacklingGame -> winner
    assert out.loc[3, "touch_class"] == "foul"  # first event a Foul
    assert out.loc[4, "player_id"] == "bc"  # co-timed partner: the partner Play is the touch, same player bc
    assert out.loc[5, "touch_class"] == "touch" and out.loc[5, "player_id"] == "solo"  # standalone BallClaiming counts


def test_pff_truth_reads_only_named_fields_and_resolves_challenge():
    ev = pd.DataFrame(
        {
            "restart_id": [1, 2],
            "setpiece_type": ["corner", "corner"],
            "taker_player_id": ["t1", "t2"],
            "first_touch_player_id": ["ft1", "ft2"],
            "first_touch_is_challenge": [False, True],
            "challenge_winner_player_id": [pd.NA, "cw2"],
            "subtype": ["CR", "PA"],
            "game_event_id": ["g1", "g2"],
        }
    )
    tidx = pd.DataFrame({"game_event_id": ["g1", "g2"], "frame_id": [100, 200]})
    out = C.pff_truth(ev, tidx).set_index("restart_id")
    assert out.loc[1, "first_touch_player_id"] == "ft1" and out.loc[1, "subtype"] == "crossed"
    assert out.loc[2, "first_touch_player_id"] == "cw2"  # a challenge resolves to the challenge winner
    assert out.loc[2, "subtype"] == "short" and out.loc[2, "anchor_frame_id"] == 200


def test_location_field_characterisation_flags_a_tracking_copy():
    events = pd.DataFrame(
        {
            "event_type": ["ShotAtGoal", "TacklingGame"],
            "field": ["shot_xy", "tackle_xy"],
            "x": [94.0, 50.0],
            "y": [34.0, 34.0],
            "player_id": ["p", "q"],
            "frame_id": [1, 2],
        }
    )
    frames = pd.DataFrame({"frame_id": [1, 2], "player_id": ["p", "q"], "x": [91.0, 50.0], "y": [34.0, 34.0]})
    out = C.location_field_characterisation(events, frames).set_index("field")
    assert not bool(out.loc["shot_xy", "is_copy"])  # 3 m away -> independent
    assert bool(out.loc["tackle_xy", "is_copy"])  # on the tracked actor -> a copy


# --- coverage, inter-run flow, course tier, census ---------------------------------------------


def test_coverage_report_counts_valued_share():
    outputs = pd.DataFrame(
        {
            "data_profile": ["p"] * 4,
            "cell": ["corner|kick_time"] * 4,
            "evidence": ["gated"] * 4,
            "cell_resolution": ["cell"] * 4,
            "value": [1.0, np.nan, 2.0, np.nan],
        }
    )
    rep = C.coverage_report(outputs)
    assert rep.iloc[0]["n"] == 4 and rep.iloc[0]["n_valued"] == 2 and rep.iloc[0]["valued_share"] == 0.5


def test_read_upstream_refuses_absent_and_commit_drift(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.read_upstream(tmp_path, "run1", expect_commit="abc")
    (tmp_path / "run1.parquet").write_bytes(pd.DataFrame({"a": [1]}).to_parquet())
    with pytest.raises(FileNotFoundError):  # sidecar missing
        C.read_upstream(tmp_path, "run1", expect_commit="abc")
    (tmp_path / "run1.json").write_text(json.dumps({"run_commit": "DIFFERENT"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="run_commit"):
        C.read_upstream(tmp_path, "run1", expect_commit="abc")
    (tmp_path / "run1.json").write_text(json.dumps({"run_commit": "abc"}), encoding="utf-8")
    table, side = C.read_upstream(tmp_path, "run1", expect_commit="abc")
    assert len(table) == 1 and side["run_commit"] == "abc"


def test_dedup_course_tier_keeps_non_overlapping():
    course = pd.DataFrame(
        {"competition": ["L", "L"], "date": ["d1", "d2"], "home_team": ["A", "C"], "away_team": ["B", "D"]}
    )
    owner = pd.DataFrame({"competition": ["L"], "date": ["d1"], "home_team": ["A"], "away_team": ["B"]})
    kept, overlap = C.dedup_course_tier(course, owner)
    assert overlap == 1 and len(kept) == 1 and kept.iloc[0]["date"] == "d2"


def test_video_census_rejects_missing_field_and_counts_only(tmp_path):
    good = [{"match_id": "m1", "source": "tv", "full_match": True, "publicly_accessible": True, "same_fixture": True}]
    p = tmp_path / "list.json"
    p.write_text(json.dumps(good), encoding="utf-8")
    prof = {"m1": DataProfile("skillcorner", "r1", "g1", "cs")}
    out = C.video_census(p, prof)
    assert int(out.iloc[0]["n_with_video"]) == 1
    p.write_text(json.dumps([{"match_id": "m1"}]), encoding="utf-8")  # missing criteria fields
    with pytest.raises(ValueError, match="criteria"):
        C.video_census(p, prof)


def test_assert_known_paired_raises_on_unpaired_opendata():
    rep = pd.DataFrame({"match_id": ["m1"], "is_opendata": [True], "paired": [False]})
    with pytest.raises(C.GiPairingCalibrationError):
        C.assert_known_paired(rep)
    C.assert_known_paired(pd.DataFrame({"match_id": ["m1"], "is_opendata": [True], "paired": [True]}))


# --- stoppage s0 and the event phase (C-PLAN-24) -----------------------------------------------


def test_stoppage_time_three_sources():
    sc = pd.DataFrame(
        {
            "restart_id": [1],
            "period_id": [1],
            "t_kick": [100.0],
            "kick_bracket_lo": [95.0],
            "is_skillcorner_derived": [True],
        }
    )
    assert C.stoppage_time(sc, None, pd.DataFrame()).iloc[0]["s0_source"] == "gi_bracket_lo"
    idsse = pd.DataFrame(
        {
            "restart_id": [1],
            "period_id": [1],
            "t_kick": [100.0],
            "kick_bracket_lo": [np.nan],
            "is_skillcorner_derived": [False],
        }
    )
    frames = pd.DataFrame(
        {
            "period_id": [1, 1, 1, 1],
            "time_seconds": [97.0, 98.0, 99.0, 101.0],
            "ball_state": ["dead", "dead", "dead", "alive"],
        }
    )
    r = C.stoppage_time(idsse, frames, pd.DataFrame())
    assert r.iloc[0]["s0_source"] == "ball_state_dead_run" and r.iloc[0]["s0"] == 97.0
    events = pd.DataFrame({"period_id": [1], "time_seconds": [90.0], "is_live": [True]})
    r2 = C.stoppage_time(idsse, None, events)
    assert r2.iloc[0]["s0_source"] == "last_live_event" and r2.iloc[0]["s0"] == 90.0


def test_dead_run_not_ending_at_kick_is_not_used():
    idsse = pd.DataFrame(
        {
            "restart_id": [1],
            "period_id": [1],
            "t_kick": [100.0],
            "kick_bracket_lo": [np.nan],
            "is_skillcorner_derived": [False],
        }
    )
    frames = pd.DataFrame(
        {
            "period_id": [1] * 4,
            "time_seconds": [80.0, 81.0, 90.0, 99.0],
            "ball_state": ["dead", "dead", "alive", "alive"],
        }
    )
    events = pd.DataFrame({"period_id": [1], "time_seconds": [95.0], "is_live": [True]})
    r = C.stoppage_time(idsse, frames, events)
    assert r.iloc[0]["s0_source"] == "last_live_event"  # the earlier dead run does not end at the kick


def test_event_phase_boundaries():
    restarts = pd.DataFrame({"period_id": [1], "s0": [50.0], "t_kick": [60.0], "kick_bracket_hi": [60.0]})
    fidx = pd.DataFrame({"period_id": [1] * 4, "time_seconds": [49.9, 50.0, 65.9, 66.1]})
    ph = C.event_phase(fidx, restarts).tolist()
    assert ph == ["open", "restart", "restart", "open"]  # s0 in, s0-0.1 out; t_kick+5.9 in, +6.1 out


# --- fidelity statistics (section 16.3) --------------------------------------------------------


def _ball_trace(times, detected, x=50.0, y=34.0, period=1, status=None):
    rows = []
    for i, (t, d) in enumerate(zip(times, detected, strict=True)):
        rows.append(
            {
                "game_id": "g",
                "period_id": period,
                "frame_id": i,
                "time_seconds": t,
                "is_ball": True,
                "x": x,
                "y": y,
                "player_id": pd.NA,
                "detected": d,
                "z_source": "measured",
                "visibility": (status[i] if status else ("detected" if d else "filled")),
            }
        )
    return pd.DataFrame(rows)


def test_extrapolated_never_counts_as_detected():
    # a ball stream that is entirely extrapolated -> detection rate 0 (C-PLAN-15)
    ball = _ball_trace([0.0, 0.1, 0.2], [False, False, False], status=["extrapolated"] * 3)
    stats = C.measure_profile_stats(ball, pd.DataFrame(columns=["t_kick", "period_id"]), None)
    det = stats[(stats["statistic"] == "ball_detection_rate") & (stats["region"] == "rest")]
    assert float(det.iloc[0]["value"]) == 0.0
    fill = stats[(stats["statistic"] == "fill_share") & (stats["region"] == "rest")]
    assert float(fill.iloc[0]["value"]) == 1.0


def test_fidelity_report_passes_matched_and_fails_mismatched():
    tgt = C.measure_profile_stats(
        _ball_trace(np.arange(0, 2, 0.1), [True] * 20), pd.DataFrame(columns=["t_kick", "period_id"]), None
    )
    same = C.measure_profile_stats(
        _ball_trace(np.arange(0, 2, 0.1), [True] * 20), pd.DataFrame(columns=["t_kick", "period_id"]), None
    )
    rep = C.fidelity_report(same, tgt)
    rest_det = rep[(rep["statistic"] == "ball_detection_rate") & (rep["region"] == "rest")]
    assert bool(rest_det.iloc[0]["passed"])
    worse = C.measure_profile_stats(
        _ball_trace(np.arange(0, 2, 0.1), [False] * 20), pd.DataFrame(columns=["t_kick", "period_id"]), None
    )
    rep2 = C.fidelity_report(worse, tgt)
    rest_det2 = rep2[(rep2["statistic"] == "ball_detection_rate") & (rep2["region"] == "rest")]
    assert not bool(rest_det2.iloc[0]["passed"])


def test_redetection_error_stats_bins_and_excludes_period_end():
    frames = pd.DataFrame(
        {
            "player_id": ["p"] * 5,
            "period_id": [1, 1, 1, 1, 1],
            "time_seconds": [0.0, 0.5, 1.0, 1.5, 2.0],
            "is_ball": [False] * 5,
            "x": [0.0, 0.0, 0.0, 0.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0],
            "visibility": ["detected", "detected", False, False, "detected"],
            "detected": [True, True, False, False, True],
        }
    )
    stats = C.redetection_error_stats(frames).set_index("bin")
    assert stats.loc["1-3s", "n"] == 1  # the 1.5 s spell (from t=0.5 to t=2.0) -> jump 3 m
    assert np.isclose(stats.loc["1-3s", "p90_m"], 3.0)


def test_degrade_reproduces_a_gap_distribution():
    rng_times = np.arange(0, 10, 0.1)
    frames = _ball_trace(rng_times, [True] * len(rng_times))
    target = {"frame_rate": 10.0, "gap_lengths": {"rest": [1.0, 1.0, 1.0]}, "noise_sd": {}}
    deg = C.degrade(frames, target, seed=1)
    # the degraded ball now carries undetected runs (gaps) that were not there before
    assert not deg[deg["is_ball"]]["detected"].all()


# --- camera-follow model (R3) ------------------------------------------------------------------


def _footprint_row(gid, pid, fid, t, half, krel=np.nan):
    """A square footprint of half-width ``half`` centred on (50, 34)."""
    cx, cy = 50.0, 34.0
    return {
        "game_id": gid,
        "period_id": pid,
        "frame_id": fid,
        "time_seconds": t,
        "cam_tl_x": cx - half,
        "cam_tl_y": cy + half,
        "cam_tr_x": cx + half,
        "cam_tr_y": cy + half,
        "cam_br_x": cx + half,
        "cam_br_y": cy - half,
        "cam_bl_x": cx - half,
        "cam_bl_y": cy - half,
        "kick_relative_s": krel,
    }


def test_fit_camera_follow_holds_two_zooms_and_pools_under_min_events():
    # two calibration matches, open play only, same ball bin: a wide (half=30) and a tight (half=10) zoom
    rows, ball_rows, phases = [], [], []
    fid = 0
    for m, half in [("A", 30.0), ("B", 10.0)]:
        for k in range(5):  # 5 frames each, one 10 s window
            rows.append(_footprint_row(m, 1, fid, float(k), half))
            ball_rows.append({"x": 50.0, "y": 34.0})
            phases.append("open")
            fid += 1
    fp = pd.DataFrame(rows)
    ball = pd.DataFrame(ball_rows)
    phase = pd.Series(phases)
    model = C.fit_camera_follow(fp, ball, phase, calibration_matches=["A", "B"], min_events=30)
    b = C._ball_bin(50.0, 34.0)
    cell = model.cell(b, "open")
    # both zoom sequences are held (not a median)
    halves = sorted({round(abs(C._seq_corners(s)[0][1][0]), 1) for s in cell})  # tr_x offset ~ half
    assert 10.0 in halves and 30.0 in halves
    # the cell has < 30 events, so it pooled within its phase (recorded)
    assert model.pooling.get((b, "open")) is not None


def test_build_footprint_chain_quick_restart_open_core_seam():
    model = _open_and_restart_model()
    # a throw-in 3 s after s0 -> core starts at s0, no setup, an open-core seam
    timeline = pd.DataFrame(
        {
            "game_id": ["g"] * 12,
            "period_id": [1] * 12,
            "frame_id": list(range(12)),
            "time_seconds": [float(t) for t in range(12)],
            "x": [50.0] * 12,
            "y": [34.0] * 12,
        }
    )
    restarts = pd.DataFrame(
        {
            "period_id": [1],
            "restart_id": [1],
            "s0": [5.0],
            "t_kick": [8.0],
            "kick_bracket_hi": [8.0],
            "s0_source": ["last_live_event"],
        }
    )
    chain = C.build_footprint_chain(model, timeline, restarts, seed=1, seg_s=10.0, stitch_k=5)
    assert "open-core" in set(chain.seams.values())
    assert chain.s0_source[1] == "last_live_event"


def _open_and_restart_model():
    seqs = {}
    # open-phase sequences in the centre bin
    b = C._ball_bin(50.0, 34.0)
    open_seq = np.array([[*([0.0, 5.0, 5.0, 5.0, 5.0, -5.0, -5.0, -5.0]), float(k)] for k in range(5)])
    restart_seq = np.array([[*([0.0, 8.0, 8.0, 8.0, 8.0, -8.0, -8.0, -8.0]), float(k - 2)] for k in range(8)])
    seqs[(b, "open")] = [open_seq for _ in range(5)]
    seqs[(b, "restart")] = [restart_seq for _ in range(5)]
    return C.CameraFollowModel(sequences=seqs, pooling={}, calibration_matches=("A",), seg_s=10.0, min_events=1)


def test_build_footprint_chain_deterministic_by_seed():
    model = _open_and_restart_model()
    timeline = pd.DataFrame(
        {
            "game_id": ["g"] * 12,
            "period_id": [1] * 12,
            "frame_id": list(range(12)),
            "time_seconds": [float(t) for t in range(12)],
            "x": [50.0] * 12,
            "y": [34.0] * 12,
        }
    )
    restarts = pd.DataFrame(
        {
            "period_id": [1],
            "restart_id": [1],
            "s0": [5.0],
            "t_kick": [8.0],
            "kick_bracket_hi": [8.0],
            "s0_source": ["x"],
        }
    )
    c1 = C.build_footprint_chain(model, timeline, restarts, seed=7)
    c2 = C.build_footprint_chain(model, timeline, restarts, seed=7)
    assert np.allclose(np.nan_to_num(c1.footprint), np.nan_to_num(c2.footprint))


def test_apply_camera_follow_masks_outside_and_keeps_inside():
    model = _open_and_restart_model()
    timeline = pd.DataFrame(
        {
            "game_id": ["g"] * 3,
            "period_id": [1] * 3,
            "frame_id": [0, 1, 2],
            "time_seconds": [0.0, 1.0, 2.0],
            "x": [50.0] * 3,
            "y": [34.0] * 3,
        }
    )
    chain = C.build_footprint_chain(
        model, timeline, pd.DataFrame(columns=["period_id", "restart_id", "s0", "t_kick", "kick_bracket_hi"]), seed=1
    )
    frames = pd.DataFrame(
        {
            "frame_id": [0, 0],
            "x": [50.0, 90.0],
            "y": [34.0, 60.0],
            "is_ball": [False, False],
            "visibility": [True, True],
        }
    )
    out = C.apply_camera_follow(model, frames, chain)
    assert bool(out.iloc[0]["visibility"]) and not bool(out.iloc[1]["visibility"])  # inside kept, outside masked


# --- off-camera replay (R2) and the footprint gate (section 16.1) -------------------------------


def test_replay_offcamera_error_displaces_the_spell():
    model = C.RedetectionModel(jump_m={"1-3s": [4.0]}, angle_rad={"1-3s": [0.0]}, calibration_matches=("A",))
    frames = pd.DataFrame(
        {
            "player_id": ["p"] * 4,
            "period_id": [1] * 4,
            "time_seconds": [0.0, 0.5, 1.0, 2.0],
            "is_ball": [False] * 4,
            "x": [0.0, 0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "detected": [True, False, False, True],
            "visibility": ["detected", False, False, "detected"],
        }
    )
    out = C.replay_offcamera_error(model, frames, seed=1)
    # the undetected (extrapolated) spell is displaced, growing to the sampled error e at the spell end
    assert out.iloc[2]["x"] != 0.0 and np.isclose(out.iloc[2]["x"], 4.0)


def test_footprint_profile_gate_passes_at_099_fails_at_098():
    # a large footprint containing every player -> share 1.0 -> pass
    prof = DataProfile("skillcorner", "r", "g", "cs")
    frames = pd.DataFrame(
        {
            "game_id": ["g"] * 2,
            "period_id": [1, 1],
            "frame_id": [1, 1],
            "is_ball": [False, False],
            "x": [50.0, 52.0],
            "y": [34.0, 35.0],
            "visibility": ["detected", "detected"],
        }
    )
    fp = pd.DataFrame([_footprint_row("g", 1, 1, 0.0, 40.0)])
    res = C.footprint_profile_gate(frames, fp, prof)
    assert res["passed"] and res["visibility_model"] == "camera_follow"
    # a tiny footprint excludes a player -> share < 0.99 -> approximation
    tiny = pd.DataFrame([{**_footprint_row("g", 1, 1, 0.0, 0.5)}])
    res2 = C.footprint_profile_gate(frames, tiny, prof)
    assert not res2["passed"] and res2["visibility_model"] == "approximation"


def test_simulate_skillcorner_penalty_semantics_goal_at_kickoff():
    ev = pd.DataFrame({"restart_id": [1, 2], "outcome": ["goal", "saved"], "rebound": [False, True]})
    out = C.simulate_skillcorner_penalty_semantics(ev, pd.DataFrame()).set_index("restart_id")
    assert out.loc[1, "goal_evidence"] == "score_at_kickoff" and out.loc[1, "has_post_goal_kickoff"]
    assert out.loc[2, "goal_evidence"] == "none" and bool(out.loc[2, "has_rebound"])


def test_ratified_constants_present():
    assert RATIFIED["cam_seg_s"] == 10.0 and RATIFIED["cam_stitch_k"] == 5


def test_tf46p2_drivers_are_ascii_only():
    # the driver ASCII gate (feedback_ruff_ascii_only_in_scripts: ruff is NOT it) -- a non-ASCII byte in a
    # driver trips the DGX run's encoding; every tf46p2 source stays ASCII.
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "scripts"
    for name in (
        "_tf46p2_common",
        "_tf46p2_score",
        "_tf46p2_pipeline",
        "tf46p2_truth",
        "tf46p2_kick_cells",
        "tf46p2_subtype_merge",
        "tf46p2_penalty",
        "tf46p2_origin",
        "tf46p2_video_census",
        "tf46p2_gate_table",
    ):
        text = (root / f"{name}.py").read_text(encoding="utf-8")
        bad = [(i + 1, repr(c)) for i, c in enumerate(text) if ord(c) > 127]
        assert not bad, f"{name}.py has non-ASCII characters: {bad[:5]}"
