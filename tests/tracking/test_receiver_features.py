"""Task 1: receiver candidate feature extractor + geometric proxy (pure, pre-pass only).

Public feature_set = POSITIONS ONLY (leakage-free on velocity-less SB360 freeze frames); the owner
feature_set adds the velocity-derived features (ball-release-direction alignment + closing speed),
which raise if asked for on a frame without velocity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._receiver import (
    ReceiverParams,
    geometric_proxy_receiver,
    receiver_candidate_features,
)

_ATT, _DEF = 1, 2


def _frame(*, with_velocity: bool) -> pd.DataFrame:
    """Passer at (50,34); ball moving +x; teammate 10 on the +x ray with defender 20 interposed."""
    rows = [
        # is_ball, id, team, gk, x, y, vx, vy
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 10.0, 0.0),  # ball, release velocity +x
        (False, 9, _ATT, False, 50.0, 34.0, 0.0, 0.0),  # passer
        (False, 10, _ATT, False, 70.0, 34.0, 2.0, 0.0),  # teammate on the +x ray (marked by 20)
        (False, 11, _ATT, False, 55.0, 50.0, 0.0, 0.0),  # teammate up-field, open
        (False, 12, _ATT, False, 55.0, 20.0, 0.0, 0.0),  # teammate down, open
        (False, 20, _DEF, False, 60.0, 34.0, 0.0, 0.0),  # defender interposed on 9->10 corridor
        (False, 21, _DEF, False, 52.0, 60.0, 0.0, 0.0),  # defender far from the lanes
        (False, 30, _DEF, True, 100.0, 34.0, 0.0, 0.0),  # defending GK
    ]
    df = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 100
    df = df.astype({"player_id": "Int64", "team_id": "Int64"})
    if not with_velocity:
        df = df.drop(columns=["vx", "vy"])
    return df


def _action():
    return pd.Series({"player_id": 9, "team_id": _ATT, "start_x": 50.0, "start_y": 34.0})


def test_public_feature_set_is_positions_only():
    feats = receiver_candidate_features(_action(), _frame(with_velocity=True), feature_set="public")
    # one row per passing-team teammate (canonical ids '10','11','12') -- passer 9 excluded
    assert set(feats["candidate_id"]) == {"10", "11", "12"}
    # POSITIONS ONLY -- no velocity-derived feature leaks into the public set
    assert set(feats.columns) == {"candidate_id", "ball_dist", "lane_pressure", "space"}
    f = feats.set_index("candidate_id")
    assert f.loc["10", "lane_pressure"] > f.loc["11", "lane_pressure"]  # 20 interposed on 9->10
    assert f.loc["10", "space"] == f["space"].min()  # 10 is the most tightly marked


def test_owner_feature_set_adds_velocity_features():
    feats = receiver_candidate_features(_action(), _frame(with_velocity=True), feature_set="owner")
    assert {"release_dir_align", "closing_speed"} <= set(feats.columns)
    f = feats.set_index("candidate_id")
    # ball release velocity is +x; teammate 10 sits on that ray -> maximal alignment
    assert f.loc["10", "release_dir_align"] == f["release_dir_align"].max()
    assert np.isfinite(f["closing_speed"]).all()


def test_owner_feature_set_raises_without_velocity():
    with pytest.raises((KeyError, ValueError)):
        receiver_candidate_features(_action(), _frame(with_velocity=False), feature_set="owner")


def test_geometric_proxy_picks_on_ray_teammate_and_needs_velocity():
    assert geometric_proxy_receiver(_action(), _frame(with_velocity=True)) == "10"
    with pytest.raises((KeyError, ValueError)):
        geometric_proxy_receiver(_action(), _frame(with_velocity=False))


def _with_direction(df: pd.DataFrame, *, att_dir: str) -> pd.DataFrame:
    """Stamp team_attacking_direction: the acting team attacks ``att_dir``, opponents the other way."""
    out = df.copy()
    out["team_attacking_direction"] = [
        pd.NA if is_ball else (att_dir if t == _ATT else ("rtl" if att_dir == "ltr" else "ltr"))
        for is_ball, t in zip(out["is_ball"], out["team_id"], strict=True)
    ]
    return out


def _reflect_frame(df: pd.DataFrame) -> pd.DataFrame:
    """180-degree point reflection into the OTHER frame handedness: positions + velocities flip, and
    the per-team attacking-direction labels swap (ADR-028/reflection)."""
    r = df.copy()
    r["x"] = 105.0 - r["x"]
    r["y"] = 68.0 - r["y"]
    for c in ("vx", "vy"):
        if c in r.columns:
            r[c] = -r[c]
    r["team_attacking_direction"] = r["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return r


@pytest.mark.parametrize("feature_set", ["public", "owner"])
def test_features_are_frame_consistent_under_away_reflection(feature_set):
    """C-H1 (review, ADR-028): the passer comes from the ACTION (action-LTR) and teammates from the
    FRAME. On a real home-attacks-right frame, an AWAY action's passer is the 180-degree reflection of
    its frame position, so mixing the two corrupts every distance/alignment feature. The extractor must
    reproject the passer into the frame convention, so an away/rtl scene yields the SAME features as its
    aligned mirror. Without the reprojection these differ -> a train/serve skew on ~50% of GS actions."""
    aligned = _with_direction(_frame(with_velocity=True), att_dir="ltr")  # acting team attacks +x: no reflection
    mirror = _reflect_frame(aligned)  # away scene: acting team attacks -x in frame (rtl)
    action = _action()  # action-LTR is per-acting-team: identical for both legs
    fa = receiver_candidate_features(action, aligned, feature_set=feature_set).set_index("candidate_id").sort_index()
    fm = receiver_candidate_features(action, mirror, feature_set=feature_set).set_index("candidate_id").sort_index()
    pd.testing.assert_frame_equal(fa, fm, atol=1e-9)


def test_proxy_is_frame_consistent_under_away_reflection():
    """C-H1: the geometric proxy has the same action-LTR passer vs frame-teammate mix. It must pick the
    same teammate on an away/rtl frame as on its aligned mirror."""
    aligned = _with_direction(_frame(with_velocity=True), att_dir="ltr")
    mirror = _reflect_frame(aligned)
    assert geometric_proxy_receiver(_action(), aligned) == geometric_proxy_receiver(_action(), mirror) == "10"


def test_extractor_is_pure():
    frame = _frame(with_velocity=True)
    before = frame.copy(deep=True)
    receiver_candidate_features(_action(), frame, feature_set="owner")
    pd.testing.assert_frame_equal(frame, before)


def test_owner_ball_less_frame_raises_no_release_direction_but_missing_columns_stays_loud():
    """Q5: two DIFFERENT-severity failures. A frame with vx/vy COLUMNS but no BALL ROW is a per-frame gap
    -> NoReleaseDirectionError (caller skips/NaNs). Missing vx/vy columns entirely is a whole-frame-set
    misconfiguration (owner routed to a velocity-less provider) -> a LOUD KeyError, never swallowed."""
    from silly_kicks.tracking._receiver import NoReleaseDirectionError

    fr = _frame(with_velocity=True)
    fr_no_ball = fr[~fr["is_ball"].to_numpy(dtype=bool)].copy()  # keep vx/vy columns, drop the ball ROW
    with pytest.raises(NoReleaseDirectionError):
        receiver_candidate_features(_action(), fr_no_ball, feature_set="owner")
    with pytest.raises(KeyError):  # missing vx/vy COLUMNS -> loud, distinct from the per-frame exception
        receiver_candidate_features(_action(), _frame(with_velocity=False), feature_set="owner")


def test_params_is_frozen():
    import dataclasses

    p = ReceiverParams()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.proxy_cone_deg = 1.0  # type: ignore[misc]
