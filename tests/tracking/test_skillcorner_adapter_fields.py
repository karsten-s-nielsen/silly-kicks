"""Task 2.3b SkillCorner tracking adapter (spec section 17.7, DL ruling): ball detection `visibility`,
frame-level possession, `in_camera_view`, and the orientation flip decided ONCE and recorded on the report so
the camera footprint cannot disagree with the frames."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import skillcorner as sk
from silly_kicks.tracking._restart_frame import resolve_frame_flips
from silly_kicks.tracking.schema import (
    FRAME_FLIPS_ATTR,
    SKILLCORNER_CAMERA_FOOTPRINT_COLUMNS,
    TrackingConversionReport,
)

# centre-origin corners of the FULL pitch -> scaled to the whole SPADL 105x68 (everyone inside).
_FULL_CAM = {"tl": (-52.5, 34.0), "tr": (52.5, 34.0), "br": (52.5, -34.0), "bl": (-52.5, -34.0)}
# centre-origin corners of the LEFT half only -> scaled to SPADL x in [0, 52.5].
_LEFT_CAM = {"tl": (-52.5, 34.0), "tr": (0.0, 34.0), "br": (0.0, -34.0), "bl": (-52.5, -34.0)}


def _row(frame, period, ts, player, team, x, y, is_gk, *, vis=True, ball_vis=True, cam=_FULL_CAM, adapter=True):
    r = {
        "match_id": "m1",
        "period": period,
        "frame": frame,
        "timestamp": ts,
        "player_id": player,
        "team_id": team,
        "is_goalkeeper": is_gk,
        "x": x,
        "y": y,
        "ball_x": 5.0,
        "ball_y": 1.0,
        "ball_z": 2.0,
        "is_visible": vis,
        "frame_rate": 10,
        "pitch_length": 105.0,
        "pitch_width": 68.0,
    }
    if adapter:
        r["ball_is_visible"] = ball_vis
        r["sc_possession_player_id"] = "311"
        r["sc_possession_group"] = "home"
        for c, (cx, cy) in cam.items():
            r[f"cam_{c}_x"] = cx
            r[f"cam_{c}_y"] = cy
    return r


def _bronze(*, p2_home_x=50.0, cam=_FULL_CAM, adapter=True, ball_vis=True, home_gk=True, n_frames=4):
    """P1 home GK left (x=-50, no flip); P2 home GK at ``p2_home_x`` (default +50 -> P2 flips)."""
    rows = []
    for f in range(n_frames):
        period = 1 if f < n_frames // 2 else 2
        ts = f * 0.1 if period == 1 else 2700.0 + (f - n_frames // 2) * 0.1
        hgk_x = -50.0 if period == 1 else p2_home_x
        agk_x = 50.0 if period == 1 else -p2_home_x
        rows += [
            _row(f, period, ts, 311, 31, hgk_x, 0.0, home_gk, ball_vis=ball_vis, cam=cam, adapter=adapter),
            _row(f, period, ts, 312, 31, -10.0, 5.0, False, ball_vis=ball_vis, cam=cam, adapter=adapter),
            _row(f, period, ts, 421, 42, agk_x, 0.0, True, vis=False, ball_vis=ball_vis, cam=cam, adapter=adapter),
            _row(f, period, ts, 422, 42, 10.0, -5.0, False, ball_vis=ball_vis, cam=cam, adapter=adapter),
        ]
    return pd.DataFrame(rows)


def test_ball_visibility_from_is_detected():
    frames, _ = sk.convert_to_frames(_bronze(ball_vis=False), home_team_id="31", output_convention="absolute_frame")
    ball = frames[frames.is_ball]
    assert (ball["visibility"] == False).all()  # noqa: E712 -- from ball_data.is_detected
    frames2, _ = sk.convert_to_frames(_bronze(ball_vis=True), home_team_id="31", output_convention="absolute_frame")
    assert (frames2[frames2.is_ball]["visibility"] == True).all()  # noqa: E712


def test_sc_possession_carried_on_every_row():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert (frames["sc_possession_player_id"].astype("string") == "311").all()
    assert (frames["sc_possession_group"].astype("string") == "home").all()
    assert str(frames["sc_possession_player_id"].dtype) == "category"
    assert "311.0" not in list(frames["sc_possession_player_id"].cat.categories)  # DL ruling 2: no ".0"


def test_in_camera_view_true_inside_full_footprint():
    frames, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert (frames["in_camera_view"] == True).all()  # noqa: E712 -- full-pitch footprint contains everyone


def test_in_camera_view_false_outside_small_footprint():
    # LEFT-half footprint; the away outfielder (centre-origin x=10 -> SPADL ~62.5) is on the right half -> outside.
    frames, _ = sk.convert_to_frames(_bronze(cam=_LEFT_CAM), home_team_id="31", output_convention="absolute_frame")
    away_out = frames[(frames.player_id == "422") & (frames.frame_id == 0)].iloc[0]
    assert bool(away_out["in_camera_view"]) is False


def test_ltr_flip_decision_recorded_asymmetric():
    _frames, report = sk.convert_to_frames(_bronze(), home_team_id="31")  # ltr
    assert report.ltr_flip_by_period == {1: False, 2: True}  # P1 already home-attacks-right; P2 flips
    assert report.game_id == "m1"


def test_absolute_frame_records_no_flips():
    _frames, report = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert report.ltr_flip_by_period == {}


def test_frames_attrs_stamp_the_applied_flip_per_period():
    # R.2 2026-10-23: the GI direction check reads the flip APPLIED to the frames (keyword first, this stamp next)
    ltr, report = sk.convert_to_frames(_bronze(), home_team_id="31")
    assert ltr.attrs[FRAME_FLIPS_ATTR] == [["m1", p, v] for p, v in sorted(report.ltr_flip_by_period.items())]
    assert ltr.attrs[FRAME_FLIPS_ATTR] == [["m1", 1, False], ["m1", 2, True]]
    absf, _ = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert absf.attrs[FRAME_FLIPS_ATTR] == [["m1", 1, False], ["m1", 2, False]]  # delivered frame: no flip applied
    assert resolve_frame_flips(ltr) == {("m1", 1): False, ("m1", 2): True}


def test_frame_flips_from_report_keys_by_match_for_both_conventions():
    _ltr, report = sk.convert_to_frames(_bronze(), home_team_id="31")
    assert sk.frame_flips_from_report(report) == {("m1", 1): False, ("m1", 2): True}
    _abs, areport = sk.convert_to_frames(_bronze(), home_team_id="31", output_convention="absolute_frame")
    assert sk.frame_flips_from_report(areport, output_convention="absolute_frame") == {
        ("m1", 1): False,
        ("m1", 2): False,
    }
    assert sk.frame_flips_from_report(areport) == {}  # an "ltr" read of an absolute report: nothing recorded


def test_frame_flips_from_report_refuses_an_unkeyable_report():
    _ltr, report = sk.convert_to_frames(_bronze(), home_team_id="31")
    with pytest.raises(ValueError, match="game_id"):
        sk.frame_flips_from_report(dataclasses.replace(report, game_id=None))
    with pytest.raises(ValueError, match="skillcorner"):
        sk.frame_flips_from_report(dataclasses.replace(report, provider="idsse"))


def test_in_camera_view_is_orientation_independent():
    # Computed before orientation -> identical in ltr and absolute_frame on the same bronze (DL e2e).
    b = _bronze(cam=_LEFT_CAM)
    ltr, _ = sk.convert_to_frames(b, home_team_id="31")
    absf, _ = sk.convert_to_frames(b, home_team_id="31", output_convention="absolute_frame")
    key = ["player_id", "frame_id", "period_id"]
    a = ltr.sort_values(key).reset_index(drop=True)
    c = absf.sort_values(key).reset_index(drop=True)
    assert (
        a["in_camera_view"].astype("string").fillna("NA").tolist()
        == c["in_camera_view"].astype("string").fillna("NA").tolist()
    )


def test_convert_camera_footprint_applies_the_recorded_flip():
    b = _bronze()
    _frames, report = sk.convert_to_frames(b, home_team_id="31")  # P2 flips
    fp = sk.convert_camera_footprint(b, report=report)
    assert list(fp.columns) == list(SKILLCORNER_CAMERA_FOOTPRINT_COLUMNS)
    assert len(fp) == 4  # one row per (game, period, frame)
    p1 = fp[fp.period_id == 1].iloc[0]
    p2 = fp[fp.period_id == 2].iloc[0]
    # P1 unflipped: tl of the full-pitch cam scales to (0, 68). P2 flipped: 105 - 0 = 105.
    assert p1["cam_tl_x"] == pytest.approx(0.0) and p2["cam_tl_x"] == pytest.approx(105.0)


def test_convert_camera_footprint_raises_on_wrong_match_or_provider():
    b = _bronze()
    bad_match = TrackingConversionReport(
        provider="skillcorner",
        total_input_frames=0,
        total_output_rows=0,
        n_periods=0,
        frame_coverage_per_period={},
        ball_out_seconds_per_period={},
        nan_rate_per_column={},
        derived_speed_rows=0,
        unrecognized_player_ids=set(),
        game_id="OTHER",
    )
    with pytest.raises(ValueError, match="does not match"):
        sk.convert_camera_footprint(b, report=bad_match)
    not_sc = TrackingConversionReport(
        provider="sportec",
        total_input_frames=0,
        total_output_rows=0,
        n_periods=0,
        frame_coverage_per_period={},
        ball_out_seconds_per_period={},
        nan_rate_per_column={},
        derived_speed_rows=0,
        unrecognized_player_ids=set(),
    )
    with pytest.raises(ValueError, match="expected 'skillcorner'"):
        sk.convert_camera_footprint(b, report=not_sc)


def test_convert_camera_footprint_no_flip_report_leaves_footprint_unflipped():
    # A report that records NO flips yields the raw-scaled footprint (P2 NOT flipped), unlike the real report
    # (P2 flipped) -- the proof that the RECORDED flip is what convert_camera_footprint applies.
    b = _bronze()
    _frames, report = sk.convert_to_frames(b, home_team_id="31")  # P2 flips
    no_flip = TrackingConversionReport(
        provider="skillcorner",
        total_input_frames=0,
        total_output_rows=0,
        n_periods=0,
        frame_coverage_per_period={},
        ball_out_seconds_per_period={},
        nan_rate_per_column={},
        derived_speed_rows=0,
        unrecognized_player_ids=set(),
        game_id="m1",
        ltr_flip_by_period={},
    )
    flipped = sk.convert_camera_footprint(b, report=report)
    unflipped = sk.convert_camera_footprint(b, report=no_flip)
    p2_flip = flipped[flipped.period_id == 2].iloc[0]["cam_tl_x"]
    p2_unflip = unflipped[unflipped.period_id == 2].iloc[0]["cam_tl_x"]
    assert p2_flip == pytest.approx(105.0) and p2_unflip == pytest.approx(0.0)


def test_backward_compatible_bronze_without_adapter_fields():
    frames, report = sk.convert_to_frames(_bronze(adapter=False), home_team_id="31", output_convention="absolute_frame")
    for c in ("sc_possession_player_id", "sc_possession_group", "in_camera_view"):
        assert c in frames.columns
        assert frames[c].isna().all()
    assert report.n_degenerate_footprint_frames == 0


def test_no_native_gk_flag_still_orients_and_records():
    # Team 31 has NO native GK flag -> derive_goalkeepers resolves one; the flip decision reads the RESOLVED GK,
    # so frames orient and the flip is still recorded (no crash, non-empty mapping).
    _frames, report = sk.convert_to_frames(_bronze(home_gk=False), home_team_id="31")
    assert set(report.ltr_flip_by_period) == {1, 2}


def test_adapter_fields_memory_is_bounded_and_row_independent():
    # R1 / B-17 / B-23: per-row <= 4 bytes (category code int8 + nullable boolean data+mask), and the category
    # dictionary is row-INDEPENDENT and <= 4 KiB per column.
    small, _ = sk.convert_to_frames(_bronze(n_frames=4), home_team_id="31", output_convention="absolute_frame")
    big, _ = sk.convert_to_frames(_bronze(n_frames=40), home_team_id="31", output_convention="absolute_frame")
    assert len(big) > len(small)
    for col in ("sc_possession_player_id", "sc_possession_group"):
        assert str(small[col].dtype) == "category"
        assert small[col].cat.codes.dtype == np.int8  # <=127 categories -> 1 byte per row
        d_small = int(small[col].cat.categories.memory_usage(deep=True))
        d_big = int(big[col].cat.categories.memory_usage(deep=True))
        assert d_small == d_big  # ROW-INDEPENDENT dictionary (same on 1x and 10x rows)
        assert d_small <= 4096  # <= 4 KiB cap
    assert str(small["in_camera_view"].dtype) == "boolean"  # nullable boolean: 1 byte data + 1 byte mask


def test_ball_detection_does_not_change_player_rows_cl17_invariance():
    # CL-17 (Chesterton / Hyrum): carrying ball detection must not change any PLAYER row -- every existing
    # detected_mask consumer filters to non-ball rows first, so identical player rows prove invariance. A
    # failure here is OD-7 (stop + report, never patch the consumer).
    with_det, _ = sk.convert_to_frames(_bronze(ball_vis=False), home_team_id="31", output_convention="absolute_frame")
    without, _ = sk.convert_to_frames(_bronze(adapter=False), home_team_id="31", output_convention="absolute_frame")
    cols = ["player_id", "frame_id", "period_id", "x", "y", "visibility", "is_goalkeeper", "is_goalkeeper_source"]
    key = ["player_id", "frame_id", "period_id"]
    a = with_det[~with_det.is_ball][cols].sort_values(key).reset_index(drop=True)
    b = without[~without.is_ball][cols].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def _cl17_variants(n_frames: int = 40):
    """The same SkillCorner synthetic match WITH ball detection carried (the Part 2 adapter) and WITHOUT it (the
    pre-adapter bronze)."""
    with_det, _ = sk.convert_to_frames(
        _bronze(ball_vis=False, n_frames=n_frames), home_team_id="31", output_convention="absolute_frame"
    )
    without, _ = sk.convert_to_frames(
        _bronze(adapter=False, n_frames=n_frames), home_team_id="31", output_convention="absolute_frame"
    )
    return with_det, without


def _cl17_actions(frames, type_name: str, team, **extra):
    from silly_kicks.spadl import config as spadlconfig

    t = float(frames["time_seconds"].iloc[len(frames) // 4])
    row = {
        "game_id": frames["game_id"].iloc[0],
        "original_event_id": "e0",
        "action_id": 0,
        "period_id": 1,
        "time_seconds": t,
        "team_id": team,
        "player_id": "311",
        "start_x": float("nan"),
        "start_y": float("nan"),
        "end_x": 50.0,
        "end_y": 34.0,
        "type_id": spadlconfig.actiontype_id[type_name],
        "result_id": 1,
        "bodypart_id": 0,
        **extra,
    }
    return pd.DataFrame([row])


def test_cl17_every_named_detected_mask_consumer_is_invariant_to_carried_ball_detection():
    # Plan Task 2.3b / CL-17 (OD-7 stop condition): drive EACH named existing consumer of player visibility /
    # detected_mask on the same SkillCorner match with and without ball detection carried, and require identical
    # output. A failure is OD-7: stop and report -- never patch the consumer (a consumer change is a retrain
    # trigger, B-21).
    import warnings

    from silly_kicks.coordination import build_coordination_signals, period_windows
    from silly_kicks.gkdv import GkdvParams
    from silly_kicks.gkdv._engine import _apply_domain
    from silly_kicks.tracking._ghost_gk import keeper_detection_mask
    from silly_kicks.tracking._gk_geometry import resolve_restart_geometry
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking.features import add_pre_shot_gk_position

    a, b = _cl17_variants()
    warnings.simplefilter("ignore")
    # 1. _gk_geometry: the tracking-GK tiers (`_tracking_gk_xy` :159, `_tracking_gk_xy_detected` :214)
    gk_act_a, gk_act_b = _cl17_actions(a, "goalkick", "31"), _cl17_actions(b, "goalkick", "31")
    pd.testing.assert_frame_equal(
        resolve_restart_geometry(gk_act_a, frames=a, distrust_native_origin=True),
        resolve_restart_geometry(gk_act_b, frames=b, distrust_native_origin=True),
    )
    # 2. tracking/utils.py :1386 -- the defending-keeper detection gate behind the pre-shot GK features
    shot_a = _cl17_actions(a, "shot", "31", defending_gk_player_id="421")
    shot_b = _cl17_actions(b, "shot", "31", defending_gk_player_id="421")
    pd.testing.assert_frame_equal(add_pre_shot_gk_position(shot_a, a), add_pre_shot_gk_position(shot_b, b))
    # 3. gkdv/_engine.py :317 -- the detected defending-keeper set of the GKDV domain (the same carrier both runs)
    keys = a[["game_id", "period_id", "frame_id"]].drop_duplicates()
    carrier = keys.assign(ball_carrier_team_id="31")
    dom_a = _apply_domain(a, carrier=carrier, goal_map=resolve_defended_goals(a), params=GkdvParams())
    dom_b = _apply_domain(b, carrier=carrier, goal_map=resolve_defended_goals(b), params=GkdvParams())
    for x, y in zip(dom_a, dom_b, strict=True):
        pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True))
    # 4. _ghost_gk.py :425 -- keeper_detection_mask on the keeper rows
    gk_a = a[a["is_goalkeeper"].astype(bool) & ~a["is_ball"].astype(bool)].sort_values(["frame_id", "player_id"])
    gk_b = b[b["is_goalkeeper"].astype(bool) & ~b["is_ball"].astype(bool)].sort_values(["frame_id", "player_id"])
    np.testing.assert_array_equal(
        keeper_detection_mask(gk_a["visibility"], provider="skillcorner"),
        keeper_detection_mask(gk_b["visibility"], provider="skillcorner"),
    )
    # 5. coordination/_signals.py :784 -- the detection-aware player slice of the coordination signals
    # (coordination refuses unoriented frames, spec 7.3: the same two variants, converted to the LTR convention)
    la, _ = sk.convert_to_frames(_bronze(ball_vis=False, n_frames=40), home_team_id="31", output_convention="ltr")
    lb, _ = sk.convert_to_frames(_bronze(adapter=False, n_frames=40), home_team_id="31", output_convention="ltr")
    sig_a = build_coordination_signals(la, windows=period_windows(la))
    sig_b = build_coordination_signals(lb, windows=period_windows(lb))
    assert repr(sig_a) == repr(sig_b)


def _bronze_periods(spec, *, n_per_period=2):
    """One block of frames per ``(period, home_gk_x, away_gk_x)``; ``None`` = that keeper's position unknown
    (NaN), so the period has no anchor from it."""
    rows, f = [], 0
    for period, hgk_x, agk_x in spec:
        for k in range(n_per_period):
            ts = (period - 1) * 2700.0 + k * 0.1
            hx = float("nan") if hgk_x is None else hgk_x
            ax = float("nan") if agk_x is None else agk_x
            rows += [
                _row(f, period, ts, 311, 31, hx, 0.0, True),
                _row(f, period, ts, 312, 31, -10.0, 5.0, False),
                _row(f, period, ts, 421, 42, ax, 0.0, True),
                _row(f, period, ts, 422, 42, 10.0, -5.0, False),
            ]
            f += 1
    return pd.DataFrame(rows)


def test_no_anchor_period_records_none_and_flips_neither_frames_nor_footprint():
    # DL ruling (spec section 17.7): a period with no keeper anchor records None; the orienter leaves it as-is
    # (with its warning) and the footprint, which only applies the RECORDED flip, is not flipped either.
    b = _bronze_periods([(1, -50.0, 50.0), (2, None, None)])
    with pytest.warns(UserWarning):
        frames, report = sk.convert_to_frames(b, home_team_id="31")
    assert report.ltr_flip_by_period[1] is False and report.ltr_flip_by_period[2] is None
    raw, _ = sk.convert_to_frames(b, home_team_id="31", output_convention="absolute_frame")
    key = ["period_id", "frame_id", "player_id"]
    p2 = frames[frames.period_id == 2].sort_values(key)["x"].to_numpy()
    p2_raw = raw[raw.period_id == 2].sort_values(key)["x"].to_numpy()
    assert len(p2) and len(p2) == len(p2_raw)  # non-vacuity: the period has rows
    np.testing.assert_array_equal(p2, p2_raw)  # not flipped
    fp = sk.convert_camera_footprint(b, report=report)
    assert fp[fp.period_id == 2].iloc[0]["cam_tl_x"] == pytest.approx(0.0)  # not flipped


def test_period_five_never_flips_frames_or_footprint():
    # DL ruling: period 5 (the shoot-out) has no defined orientation; it is recorded False even when its keeper
    # geometry would flip a regular period, and neither the frames nor the footprint flip.
    b = _bronze_periods([(1, -50.0, 50.0), (5, 50.0, -50.0)])
    frames, report = sk.convert_to_frames(b, home_team_id="31")
    assert report.ltr_flip_by_period[5] is False
    raw, _ = sk.convert_to_frames(b, home_team_id="31", output_convention="absolute_frame")
    key = ["period_id", "frame_id", "player_id"]
    p5 = frames[frames.period_id == 5].sort_values(key)["x"].to_numpy()
    p5_raw = raw[raw.period_id == 5].sort_values(key)["x"].to_numpy()
    assert len(p5) and len(p5) == len(p5_raw)  # non-vacuity
    np.testing.assert_array_equal(p5, p5_raw)
    fp = sk.convert_camera_footprint(b, report=report)
    assert fp[fp.period_id == 5].iloc[0]["cam_tl_x"] == pytest.approx(0.0)


def test_convert_camera_footprint_refuses_a_report_without_a_game_id():
    # A report that cannot show it belongs to this match must not orient its footprint (C2-IMPL-18).
    b = _bronze()
    no_id = TrackingConversionReport(
        provider="skillcorner",
        total_input_frames=0,
        total_output_rows=0,
        n_periods=0,
        frame_coverage_per_period={},
        ball_out_seconds_per_period={},
        nan_rate_per_column={},
        derived_speed_rows=0,
        unrecognized_player_ids=set(),
        game_id=None,
    )
    with pytest.raises(ValueError, match="does not match"):
        sk.convert_camera_footprint(b, report=no_id)
