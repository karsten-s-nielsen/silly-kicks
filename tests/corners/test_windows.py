"""TF-46 add_corner_windows: TF-79 timing + winner + diagnostics."""

from __future__ import annotations

import pandas as pd

from silly_kicks.corners import CORNER_WINDOW_COLUMNS, add_corner_windows


def _corner(source="spadl_corner", t_kick=100.0, period=1):
    return pd.DataFrame(
        {
            "game_id": ["g1"],
            "corner_id": [0],
            "period_id": [period],
            "t_kick": [t_kick],
            "attacking_team_id": ["tA"],
            "corner_side": pd.Series(["left"], dtype="category"),
            "kick_x": [104.5],
            "kick_y": [1.0],
            "corner_kind": pd.Series(["direct"], dtype="category"),
            "corner_source": pd.Series([source], dtype="category"),
        }
    )


def _frames():
    return pd.DataFrame(
        {
            "frame_id": [5, 5],
            "period_id": [1, 1],
            "time_seconds": [100.5, 100.5],
            "player_id": ["pW", "pX"],
            "team_id": ["tW", "tX"],
        }
    )


def _contacts(frame_id=5, t=100.5, player="pW", period=1):
    return pd.DataFrame(
        {
            "period_id": [period],
            "frame_id": [frame_id],
            "time_seconds": [t],
            "player_id": [player],
            "contact_source": pd.Series(["velocity_break"], dtype="object"),
        }
    )


def _no_contacts():
    return pd.DataFrame(
        {
            "period_id": pd.Series([], dtype="int64"),
            "frame_id": pd.Series([], dtype="int64"),
            "time_seconds": pd.Series([], dtype="float64"),
            "player_id": pd.Series([], dtype="object"),
            "contact_source": pd.Series([], dtype="object"),
        }
    )


def test_target_uses_first_contact_plus_lead_when_resolved():
    out = add_corner_windows(_corner(), _frames(), _contacts())
    r = out.iloc[0]
    assert r["target_snapshot_source"] == "first_contact"
    assert r["first_contact_time"] == 100.5
    assert r["t_target"] == min(100.5 + 1.0, 100.0 + 2.0)  # 101.5
    assert r["t_init"] == 98.0
    assert r["t_outcome"] == 106.0


def test_winner_team_resolved_from_frames():
    out = add_corner_windows(_corner(), _frames(), _contacts(player="pW"))
    r = out.iloc[0]
    assert r["winner_player_id"] == "pW"
    assert r["winner_team_id"] == "tW"


def test_fallback_to_kick_plus_2s_when_no_contact():
    out = add_corner_windows(_corner(), _frames(), _no_contacts())
    r = out.iloc[0]
    assert r["target_snapshot_source"] == "kick_plus_2s"
    assert r["t_target"] == 102.0
    assert pd.isna(r["winner_team_id"])  # no GI fallback
    assert pd.isna(r["winner_player_id"])
    assert pd.isna(r["first_contact_time"])


def test_sb360_no_contacts_winner_is_na_source_sb360_event():
    out = add_corner_windows(_corner(source="sb360_event"), _frames(), _no_contacts())
    r = out.iloc[0]
    assert r["target_snapshot_source"] == "sb360_event"
    assert pd.isna(r["winner_team_id"])
    assert pd.isna(r["first_contact_time"])


def test_first_possession_diagnostic_column_emitted():
    out = add_corner_windows(_corner(), _frames(), _contacts())
    assert "first_possession_time" in out.columns


def test_gi_populates_first_possession_time():
    gi = pd.DataFrame({"game_id": ["g1"], "period_id": [1], "time_seconds": [100.0]})
    out = add_corner_windows(_corner(), _frames(), _contacts(), gi=gi)
    assert out.iloc[0]["first_possession_time"] == 100.0


def test_all_window_columns_present_and_typed():
    out = add_corner_windows(_corner(), _frames(), _contacts())
    for col in CORNER_WINDOW_COLUMNS:
        assert col in out.columns


def test_pure_does_not_mutate_inputs():
    corners, frames, contacts = _corner(), _frames(), _contacts()
    cb, fb, kb = corners.copy(deep=True), frames.copy(deep=True), contacts.copy(deep=True)
    out = add_corner_windows(corners, frames, contacts)
    pd.testing.assert_frame_equal(corners, cb)
    pd.testing.assert_frame_equal(frames, fb)
    pd.testing.assert_frame_equal(contacts, kb)
    assert out is not corners


def test_nan_t_kick_keeps_row_with_nan_window():
    # A derived (events-only) corner has a NaN t_kick (spec section 9.4): the row is KEPT with NaN
    # window outputs and no target source (ADR-042), never dropped and never crashed.
    out = add_corner_windows(_corner(t_kick=float("nan")), _frames(), _no_contacts())
    assert len(out) == 1
    r = out.iloc[0]
    assert pd.isna(r["t_init"]) and pd.isna(r["t_target"]) and pd.isna(r["t_outcome"])
    assert pd.isna(r["target_snapshot_source"])
