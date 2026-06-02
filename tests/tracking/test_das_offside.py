"""DAS carrier forwarding (Phase 0c, corrected 4.4.1).

Forwarding ``player_in_possession_col`` to accessible-space exempts the passer from the
offside mask (accessible-space deletes offside attackers — "treats them like air"). Effect:

* carrier ONSIDE  -> value-neutral (nothing to exempt) — ``test_onside_*``
* carrier OFFSIDE -> DAS CHANGES (the on-ball carrier is no longer mis-flagged offside and
  deleted) — ``test_offside_carrier_forwarding_changes_das``

The original 4.2.0 "value-neutral on real data" claim was wrong: its fixture placed the
carrier clearly onside, so it never exercised the offside path. On real matches the carrier
is frequently tracked just ahead of the ball/offside line, so 4.2.0 changed DAS (a correctness
fix — see ADR-012 amendment + CHANGELOG 4.4.1).
"""

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
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
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


def test_onside_carrier_forwarding_is_value_neutral_and_silences_warning():
    """When the carrier is ONSIDE, forwarding it is value-neutral (nothing to exempt).

    This is the ONLY regime in which forwarding is value-neutral; it does NOT generalize to
    real matches (see test_offside_carrier_forwarding_changes_das). It still verifies the
    accessible-space per-call offside warning is silenced.
    """
    frames = _frames_with_carrier()  # carrier placed clearly onside (x=30, defenders at x=70+)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r_on = get_das(frames, player_in_possession_col="ball_carrier_player_id")
    assert not any("player_in_possession_col should be set" in str(x.message) for x in w), (
        "library offside warning not silenced"
    )
    r_off = get_das(frames, player_in_possession_col=None)
    np.testing.assert_allclose(r_on["AS"].to_numpy(), r_off["AS"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(r_on["DAS"].to_numpy(), r_off["DAS"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True)


def _frames_with_offside_carrier() -> pd.DataFrame:
    """Home in possession attacking +x; the on-ball carrier (10) sits BEYOND the offside line.

    Offside line = max(2nd-last defender norm-x, ball norm-x) = max(60, 75) = 75. Carrier 10 is
    at x=80 (5 m ahead of the ball at 75) -> would be flagged offside and DELETED unless exempted
    via player_in_possession_col. Teammate 11 (x=40) and both Away defenders (60, 70) are onside.
    This reproduces, minimally, the real-match situation where the on-ball carrier is tracked just
    ahead of the ball (ADR-012 amendment).
    """
    rng = np.random.default_rng(0)
    rows = []
    for fid in range(5):
        for pid, tid, x in [
            (10, "Home", 80.0),  # carrier, on the ball, beyond the offside line
            (11, "Home", 40.0),  # onside teammate
            (20, "Away", 60.0),  # 2nd-last defender (sets the offside line)
            (21, "Away", 70.0),
        ]:
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": fid,
                    "player_id": pid,
                    "team_id": tid,
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


def test_offside_carrier_forwarding_changes_das():
    """Forwarding the carrier CHANGES DAS when the carrier would otherwise be offside.

    Locks the 4.2.0 correctness fix: without the passer exemption (4.1.1 behavior, ppc=None)
    accessible-space mis-flags the on-ball carrier as offside and deletes them ("treats like
    air"), distorting AS/DAS; with it (ppc=carrier) the carrier is retained. The two MUST differ
    — this is the path the original 4.2.0 "value-neutral" test never exercised.
    """
    frames = _frames_with_offside_carrier()
    r_on = get_das(frames, player_in_possession_col="ball_carrier_player_id")
    r_off = get_das(frames, player_in_possession_col=None)
    das_on = r_on["DAS"].to_numpy()
    das_off = r_off["DAS"].to_numpy()
    # Materiality: the exemption must change DAS on at least one frame (not value-neutral here).
    assert not np.allclose(das_on, das_off, rtol=1e-6, atol=1e-6, equal_nan=True), (
        "carrier offside-exemption produced no DAS change — the offside path was not exercised"
    )
    assert np.nanmax(np.abs(das_on - das_off)) > 1.0, "expected a material (>1 m^2) DAS shift"


def test_arrow_backed_team_columns_dont_crash():
    """pyarrow-backed string team columns (newer-pandas default) must not break get_das.

    accessible-space indexes the team columns 2-D (offside path); _prepare_frames coerces
    them to numpy object. Regression for the PR #87 CI failure on py3.11/3.12 (numpy 2.4).
    """
    pytest.importorskip("pyarrow")
    frames = _frames_with_carrier()
    for c in ("team_id", "team_in_possession", "player_id"):
        frames[c] = frames[c].astype("string[pyarrow]")
    result = get_das(frames, player_in_possession_col="ball_carrier_player_id")
    assert "DAS" in result.columns and result["DAS"].notna().any()


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
