"""Task 2: played-pass extraction + orientation (`scripts/_rq_corpus.py`)."""

from __future__ import annotations

from scripts import _rq_corpus as C


def test_to_frame_coords_reflects_away_only():
    assert C.to_frame_coords(30.0, 20.0, attacks_rtl=False) == (30.0, 20.0)
    # away attacks x=105 in action-LTR -> point-reflect into home-attacks-right frame
    assert C.to_frame_coords(30.0, 20.0, attacks_rtl=True) == (75.0, 48.0)  # (105-30, 68-20)


def test_completed_pass_target_is_receiver_frame_position(mini_actions, mini_frames):
    out = C.extract_played_passes(mini_actions, mini_frames)
    row = out[out["action_id"] == 0].iloc[0]
    assert row["target_source"] == "receiver" and bool(row["is_completed"])
    assert not bool(row["is_fail"]) and not bool(row["is_cross"])
    # receiver is player 11 at frame position (58, 34); home attacks ltr -> no reflection
    assert abs(row["target_x"] - 58.0) < 1e-6 and abs(row["target_y"] - 34.0) < 1e-6


def test_failed_pass_target_is_end_xy(mini_actions, mini_frames):
    out = C.extract_played_passes(mini_actions, mini_frames)
    row = out[out["action_id"] == 1].iloc[0]
    assert row["target_source"] == "end_xy" and bool(row["is_fail"])
    assert abs(row["target_x"] - 80.0) < 1e-6  # end_x, home -> no reflection


def test_both_passes_extracted(mini_actions, mini_frames):
    out = C.extract_played_passes(mini_actions, mini_frames)
    assert set(out["action_id"]) == {0, 1}


def test_away_pass_passer_and_end_reprojected(mini_frames):
    """ADR-028 backstop: an AWAY (team 2, 'rtl') action-LTR pass reprojects into frame coords
    (start_x=30 -> 105-30=75); a driver that skipped this would place the passer mirror-wrong."""
    import pandas as pd

    from silly_kicks.spadl import config as spc

    away = pd.DataFrame(
        {
            "game_id": [1],
            "action_id": [0],
            "period_id": [1],
            "time_seconds": [1.0],
            "team_id": [2],
            "player_id": [20],
            "type_id": [spc.actiontype_id["pass"]],
            "result_id": [spc.result_id["fail"]],
            "bodypart_id": [0],
            "start_x": [30.0],
            "start_y": [34.0],
            "end_x": [50.0],
            "end_y": [34.0],
        }
    )
    row = C.extract_played_passes(away, mini_frames).iloc[0]
    assert abs(row["passer_x"] - 75.0) < 1e-6 and abs(row["passer_y"] - 34.0) < 1e-6  # (105-30, 68-34)
    assert row["target_source"] == "end_xy" and abs(row["target_x"] - 55.0) < 1e-6  # (105-50)
