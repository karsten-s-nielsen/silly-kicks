"""DAS carrier forwarding / offside-passer exclusion (native engine, ADR-107).

Forwarding ``player_in_possession_col`` exempts the passer (ball carrier) from the offside
mask. Effect:

* carrier ONSIDE  -> value-neutral (nothing to exempt) -- ``test_onside_*``
* carrier OFFSIDE -> DAS CHANGES (the on-ball carrier is no longer dropped as offside) --
  ``test_offside_carrier_forwarding_changes_das``

Native DAS resolves the attacking direction from the GoalMap (keeper geometry, ADR-055), so
both teams carry a keeper (Home near x=0 -> attacks x=105; Away keeper near x=105). The keeper
positions are chosen NOT to disturb the offside line the fixture sets up.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from silly_kicks.tracking._das import get_das


def _frames_with_carrier() -> pd.DataFrame:
    """Home attacks +x, carrier=10 placed clearly ONSIDE (deep, x=30). Keepers at the ends."""
    rows = []
    rng = np.random.default_rng(0)
    for fid in range(5):
        for pid, tid, x, gk in [
            ("hgk", "Home", 5.0, True),
            (10, "Home", 30.0, False),
            (11, "Home", 35.0, False),
            (20, "Away", 70.0, False),
            (21, "Away", 75.0, False),
            ("agk", "Away", 100.0, True),
        ]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
                    "is_goalkeeper": gk,
                    "x": x + rng.normal(0, 0.5),
                    "y": 34.0 + rng.normal(0, 1),
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "team_in_possession": "Home",
                    "ball_carrier_player_id": 10,
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "player_id": "ball",
                "team_id": None,
                "is_goalkeeper": False,
                "x": 30.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "team_in_possession": "Home",
                "ball_carrier_player_id": 10,
            }
        )
    return pd.DataFrame(rows)


def _frames_with_offside_carrier() -> pd.DataFrame:
    """Home attacks +x; the on-ball carrier (10) sits BEYOND the offside line.

    Offside line ~ max(2nd-last Away outfield defender x, ball x) = max(70, 75) = 75. Carrier 10
    at x=80 (ahead of the ball) is offside and dropped unless exempted via the carrier column. The
    Away keeper (x=104) is the DEEPEST defender, so the 2nd-last outfield defender is still 70.
    """
    rng = np.random.default_rng(0)
    rows = []
    for fid in range(5):
        for pid, tid, x, gk in [
            ("hgk", "Home", 5.0, True),
            (10, "Home", 80.0, False),  # carrier, on the ball, beyond the offside line
            (11, "Home", 40.0, False),  # onside teammate
            (20, "Away", 60.0, False),  # 2nd-last outfield defender (sets the offside line)
            (21, "Away", 70.0, False),
            ("agk", "Away", 104.0, True),  # deepest defender (keeper)
        ]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
                    "is_goalkeeper": gk,
                    "x": x + rng.normal(0, 0.3),
                    "y": 34.0 + rng.normal(0, 1),
                    "vx": 0.0,
                    "vy": 0.0,
                    "is_ball": False,
                    "team_in_possession": "Home",
                    "ball_carrier_player_id": 10,
                }
            )
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "player_id": "ball",
                "team_id": None,
                "is_goalkeeper": False,
                "x": 75.0,
                "y": 34.0,
                "vx": 0.0,
                "vy": 0.0,
                "is_ball": True,
                "team_in_possession": "Home",
                "ball_carrier_player_id": 10,
            }
        )
    return pd.DataFrame(rows)


def test_onside_carrier_forwarding_is_value_neutral():
    """A carrier that is already ONSIDE has nothing to exempt, so forwarding it is value-neutral."""
    frames = _frames_with_carrier()
    r_on = get_das(frames, player_in_possession_col="ball_carrier_player_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the no-carrier one-time UserWarning is expected here
        r_off = get_das(frames, player_in_possession_col=None)
    np.testing.assert_allclose(r_on["AS"].to_numpy(), r_off["AS"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(r_on["DAS"].to_numpy(), r_off["DAS"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_offside_carrier_forwarding_changes_das():
    """Forwarding the carrier CHANGES DAS when the carrier would otherwise be dropped as offside."""
    frames = _frames_with_offside_carrier()
    r_on = get_das(frames, player_in_possession_col="ball_carrier_player_id")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_off = get_das(frames, player_in_possession_col=None)
    das_on = r_on["DAS"].to_numpy()
    das_off = r_off["DAS"].to_numpy()
    assert not np.allclose(das_on, das_off, rtol=1e-6, atol=1e-6, equal_nan=True), (
        "carrier offside-exemption produced no DAS change -- the offside path was not exercised"
    )
    assert np.nanmax(np.abs(das_on - das_off)) > 1.0, "expected a material (>1 m^2) DAS shift"


def test_arrow_backed_team_columns_dont_crash():
    """pyarrow-backed string team columns (newer-pandas default) must not break get_das."""
    import pytest

    pytest.importorskip("pyarrow")
    frames = _frames_with_carrier()
    for c in ("team_id", "team_in_possession", "player_id"):
        frames[c] = frames[c].astype("string[pyarrow]")
    result = get_das(frames, player_in_possession_col="ball_carrier_player_id")
    assert "DAS" in result.columns and result["DAS"].notna().any()


def test_explicit_missing_carrier_col_raises():
    frames = _frames_with_carrier().drop(columns=["ball_carrier_player_id"])
    import pytest

    with pytest.raises(ValueError, match="not found"):
        get_das(frames, player_in_possession_col="nope_col")


def test_default_missing_carrier_degrades_gracefully():
    frames = _frames_with_carrier().drop(columns=["ball_carrier_player_id"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = get_das(frames)  # default name absent -> proceed without passer exclusion
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
    assert len(sk) == 1, f"expected exactly one silly-kicks warning, got {len(sk)}"


def test_all_deadball_linked_subset_degrades_with_clear_message():
    from silly_kicks.tracking.features import add_das

    frames = _frames_with_carrier()
    # Make frames 2-4 dead-ball; keep 0-1 alive so the FULL-frame direction resolves.
    frames.loc[frames["frame_id"] >= 2, "team_in_possession"] = np.nan
    actions = pd.DataFrame(
        {
            "game_id": [1, 1],
            "action_id": [1, 2],
            "period_id": [1, 1],
            "team_id": ["Home", "Home"],
            "player_id": [10, 11],
            "start_x": [30.0, 35.0],
            "start_y": [34.0, 34.0],
        }
    )
    links = pd.DataFrame(
        {
            "action_id": [1, 2],
            "frame_id": [3, 4],
            "time_offset_seconds": [0.0, 0.0],
            "n_candidate_frames": [1, 1],
            "link_quality_score": [1.0, 1.0],
        }
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = add_das(actions, frames, links=links)
    assert out["das_team"].isna().all()  # NaN-degraded
    assert any("dead-ball" in str(x.message) for x in w), "clear dead-ball message expected"
