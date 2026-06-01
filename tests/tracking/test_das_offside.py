"""Phase 0c — DAS carrier forwarding: value-neutral + warning silenced + validation."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("accessible_space")

from silly_kicks.tracking._das import get_das


def _frames_with_carrier() -> pd.DataFrame:
    """4 players + ball, possession=Home, carrier=10 placed clearly onside."""
    rows = []
    rng = np.random.default_rng(0)
    for fid in range(5):
        for pid, tid, x in [
            (10, "Home", 30.0),
            (11, "Home", 35.0),
            (20, "Away", 70.0),
            (21, "Away", 75.0),
        ]:
            rows.append(
                {"game_id": 1, "period_id": 1, "frame_id": fid, "player_id": pid,
                 "team_id": tid, "x": x + rng.normal(0, 0.5), "y": 34.0 + rng.normal(0, 1),
                 "vx": 0.0, "vy": 0.0, "is_ball": False, "team_in_possession": "Home",
                 "ball_carrier_player_id": 10}
            )
        rows.append(
            {"game_id": 1, "period_id": 1, "frame_id": fid, "player_id": "ball",
             "team_id": None, "x": 30.0, "y": 34.0, "vx": 0.0, "vy": 0.0,
             "is_ball": True, "team_in_possession": "Home", "ball_carrier_player_id": 10}
        )
    return pd.DataFrame(rows)


def test_forwarding_is_value_neutral_and_silences_warning():
    frames = _frames_with_carrier()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r_on = get_das(frames, player_in_possession_col="ball_carrier_player_id")
    assert not any(
        "player_in_possession_col should be set" in str(x.message) for x in w
    ), "library offside warning not silenced"
    r_off = get_das(frames, player_in_possession_col=None)
    np.testing.assert_allclose(
        r_on["AS"].to_numpy(), r_off["AS"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True
    )
    np.testing.assert_allclose(
        r_on["DAS"].to_numpy(), r_off["DAS"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_explicit_missing_carrier_col_raises():
    frames = _frames_with_carrier().drop(columns=["ball_carrier_player_id"])
    with pytest.raises(ValueError, match="not found"):
        get_das(frames, player_in_possession_col="nope_col")


def test_default_missing_carrier_degrades_gracefully():
    frames = _frames_with_carrier().drop(columns=["ball_carrier_player_id"])
    result = get_das(frames)  # default name absent -> degrade, no crash
    assert "DAS" in result.columns and "AS" in result.columns


def test_no_carrier_emits_one_time_silly_kicks_warning(monkeypatch):
    import silly_kicks.tracking._das as das_mod

    monkeypatch.setattr(das_mod, "_OFFSIDE_WARNED", False)
    frames = _frames_with_carrier().drop(columns=["ball_carrier_player_id"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        get_das(frames)  # default name absent -> ppc None
        get_das(frames)  # second call must NOT re-warn (one-time)
    sk = [x for x in w if "no ball-carrier column" in str(x.message)]
    lib = [x for x in w if "player_in_possession_col should be set" in str(x.message)]
    assert len(sk) == 1, f"expected exactly one silly-kicks warning, got {len(sk)}"
    assert len(lib) == 0, "library per-call offside warning must be suppressed"


def test_all_deadball_linked_subset_degrades_with_clear_message():
    from silly_kicks.tracking.features import add_das

    frames = _frames_with_carrier()
    # Make frames 2-4 dead-ball; keep 0-1 alive so the FULL-frame _pin guard passes.
    frames.loc[frames["frame_id"] >= 2, "team_in_possession"] = np.nan
    actions = pd.DataFrame(
        {"game_id": [1, 1], "action_id": [1, 2], "period_id": [1, 1],
         "team_id": ["Home", "Home"], "player_id": [10, 11],
         "start_x": [30.0, 35.0], "start_y": [34.0, 34.0]}
    )
    links = pd.DataFrame(
        {"action_id": [1, 2], "frame_id": [3, 4], "time_offset_seconds": [0.0, 0.0],
         "n_candidate_frames": [1, 1], "link_quality_score": [1.0, 1.0]}
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = add_das(actions, frames, links=links)
    assert out["das_team"].isna().all()  # NaN-degraded
    assert any("dead-ball" in str(x.message) for x in w), "clear dead-ball message expected"
