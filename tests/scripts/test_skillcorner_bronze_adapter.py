"""Task 2.3b loader: `_skillcorner_bronze` preserves ball detection, frame-level possession (canonical id,
no ".0"), and the footprint corners from the raw record shape (spec section 17.7, DL rulings C-PLAN-15 / 2)."""

from __future__ import annotations

import math

import pandas as pd

from scripts._loader_pining import _skillcorner_bronze
from tests.tracking._skillcorner_raw_shape import meta, null_frame, raw_record


def _bronze(records) -> pd.DataFrame:
    return _skillcorner_bronze(records, meta(), match_id="m1")


def test_ball_detection_preserved_true_false_null():
    assert set(_bronze([raw_record(1, 1, 0.1, ball_detected=True)])["ball_is_visible"]) == {True}
    assert set(_bronze([raw_record(1, 1, 0.1, ball_detected=False)])["ball_is_visible"]) == {False}
    out = _bronze([raw_record(1, 1, 0.1, ball_detected=None)])
    assert out["ball_is_visible"].isna().all()


def test_possession_id_is_canonical_string_without_dot_zero():
    # A JSON float id (366.0) must canonicalise to "366", never "366.0" (DL ruling 2).
    out = _bronze([raw_record(1, 1, 0.1, poss_player_id=366.0)])
    assert list(out["sc_possession_player_id"].unique()) == ["366"]
    assert list(out["sc_possession_group"].unique()) == ["attack"]


def test_null_possession_and_footprint_carry_none():
    out = _bronze([null_frame(0, 1)])
    assert out["sc_possession_player_id"].isna().all()
    assert out["sc_possession_group"].isna().all()
    assert out["cam_tl_x"].isna().all() and out["cam_br_y"].isna().all()
    assert out["ball_is_visible"].isna().all()  # is_detected null


def test_footprint_corners_mapped_from_projection_keys():
    out = _bronze([raw_record(1, 1, 0.1)]).iloc[0]
    assert out["cam_tl_x"] == -52.5 and out["cam_tl_y"] == 34.0
    assert out["cam_tr_x"] == 52.5 and out["cam_tr_y"] == 34.0
    assert out["cam_br_x"] == 52.5 and out["cam_br_y"] == -34.0
    assert out["cam_bl_x"] == -52.5 and out["cam_bl_y"] == -34.0


def test_absent_adapter_keys_give_none():
    # A record WITHOUT the possession / footprint keys (pre-adapter feed) -> None, never a crash.
    rec = {
        "frame": 1,
        "period": 1,
        "timestamp": 0.1,
        "ball_data": {"x": 1.0, "y": 2.0, "z": 0.5},  # no is_detected
        "player_data": [{"player_id": 366, "x": -10.0, "y": 0.0, "is_detected": True}],
    }
    out = _bronze([rec]).iloc[0]
    assert out["ball_is_visible"] is None
    assert out["sc_possession_player_id"] is None and out["sc_possession_group"] is None
    assert out["cam_tl_x"] is None or math.isnan(out["cam_tl_x"])
