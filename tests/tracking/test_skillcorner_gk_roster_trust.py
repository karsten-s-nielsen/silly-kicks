"""SkillCorner GK identification -- trust the native roster, don't re-derive per batch (CR 2026-06-30).

The bug: convert_to_frames discarded the clean native roster is_goalkeeper and re-derived positionally
via derive_goalkeepers. Stable on a full match, but on a small (250-frame) window a transiently
goal-parked outfielder gets flagged; across the lakehouse's per-batch builds the union reached ~15
"keepers"/team. Fix: trust the native roster GK; derive only as a fallback when it is absent.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.tracking import skillcorner as sk


def _bronze_row(frame, period, ts, player, team, x, y, is_gk, is_vis=True):
    return {
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
        "is_visible": is_vis,
        "frame_rate": 10,
    }


def _bronze_with_parked_outfielder(n_frames=20, home_gk=True):
    """home 31: GK 311 at own goal + non-GK 312 PARKED in the opponent penalty area (centre-origin
    x=42.5 -> SPADL 95, in PA) -- positional derive WOULD mis-flag 312. away 42: GK 421 + midfielder 422.
    ``home_gk`` toggles 311's native roster flag (False -> home has no native GK -> derive fallback)."""
    rows = []
    for f in range(n_frames):
        rows += [
            _bronze_row(f, 1, f * 0.1, 311, 31, -50.0, 0.0, home_gk),  # home GK, own goal (SPADL ~2.5)
            _bronze_row(f, 1, f * 0.1, 312, 31, 42.5, 0.0, False),  # home non-GK PARKED opp PA (SPADL 95)
            _bronze_row(f, 1, f * 0.1, 421, 42, 50.0, 0.0, True),  # away GK, own goal (SPADL ~102.5)
            _bronze_row(f, 1, f * 0.1, 422, 42, 0.0, 5.0, False),  # away non-GK midfield
        ]
    return pd.DataFrame(rows)


def _gk_by_team(frames):
    p = frames[~frames["is_ball"].astype(bool)]
    gk = p[p["is_goalkeeper"].astype(bool)]
    return {t: set(s["player_id"]) for t, s in gk.groupby("team_id")}


def test_trusts_native_roster_not_positional_derive():
    # the parked outfielder 312 sits in a penalty area all window -> positional derive would flag it.
    # Trusting the roster, only the native GKs are flagged.
    frames, _ = sk.convert_to_frames(_bronze_with_parked_outfielder(), home_team_id="31")
    assert _gk_by_team(frames) == {"31": {"311"}, "42": {"421"}}  # 312 NOT flagged
    p = frames[~frames["is_ball"].astype(bool)]
    assert (p["is_goalkeeper_source"] == "native").all()  # trusted, not derived


def test_skillcorner_gk_batch_invariant():
    # a 5-frame slice must yield the SAME is_goalkeeper set as the full window (roster is batch-invariant).
    full = _bronze_with_parked_outfielder(n_frames=20)
    window = full[full["frame"] < 5]
    f_full, _ = sk.convert_to_frames(full, home_team_id="31")
    f_window, _ = sk.convert_to_frames(window, home_team_id="31")
    assert _gk_by_team(f_full) == _gk_by_team(f_window) == {"31": {"311"}, "42": {"421"}}


def test_derives_when_native_absent():
    # home has NO native GK -> fall back to derivation for home only; away still trusts its roster.
    frames, report = sk.convert_to_frames(_bronze_with_parked_outfielder(home_gk=False), home_team_id="31")
    p = frames[~frames["is_ball"].astype(bool)]
    home_gk = p[(p["team_id"] == "31") & p["is_goalkeeper"].astype(bool)]
    assert len(set(home_gk["player_id"])) >= 1  # derived a fallback GK for home
    assert (home_gk["is_goalkeeper_source"] == "derived").all()
    away = p[(p["team_id"] == "42") & p["is_goalkeeper"].astype(bool)]
    assert set(away["player_id"]) == {"421"} and (away["is_goalkeeper_source"] == "native").all()
    assert report.n_teams_gk_derived == 1  # only home fell back


def test_s2_guard_warns_and_counts_on_implausible_gk_count():
    # native roster flags 3 GKs for home (>2 -> implausible) -> warn + countable report field.
    b = _bronze_with_parked_outfielder()
    b.loc[b["player_id"] == 312, "is_goalkeeper"] = True  # 311 + 312
    extra = b[b["player_id"] == 312].copy()
    extra["player_id"] = 313  # a 3rd home GK flag
    b = pd.concat([b, extra], ignore_index=True)
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _frames, report = sk.convert_to_frames(b, home_team_id="31")
    assert report.n_implausible_gk_teams >= 1
    assert any("implausible" in str(x.message).lower() or "GK count" in str(x.message) for x in w)
