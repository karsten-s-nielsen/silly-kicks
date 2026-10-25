"""Task 2.4: the F3 overhead gate conditionalisation + the attribution-margin columns (spec sections 14.1,
8, 17.4). Unit tests on `_attribute` / `_attribution_margins`; both sides of every bar."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ball_contact import BallContactParams, _attribute, _attribution_margins, detect_ball_contacts

_P = BallContactParams()
_BX = np.array([50.0])
_BY = np.array([34.0])


def _players(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


# ---- F3 overhead gate (spec 14.1) ----


def test_overhead_gate_only_when_active_and_measured():
    players = _players(player_id=["A"], x=[50.0], y=[34.0])
    high = np.array([3.0])  # above contact_max_z_m (2.6)
    # measured z + gate active -> overhead, not a contact
    assert pd.isna(_attribute(0, _BX, _BY, high, np.array(["measured"]), players, _P, overhead_active=True)[0])
    # estimated z + gate active -> NOT gated (z provenance not trustworthy)
    assert _attribute(0, _BX, _BY, high, np.array(["estimated"]), players, _P, overhead_active=True)[0] == "A"
    # measured z but gate INACTIVE (CL-3: empty table) -> not gated
    assert _attribute(0, _BX, _BY, high, np.array(["measured"]), players, _P, overhead_active=False)[0] == "A"
    # below the ceiling -> always a contact
    low = np.array([1.0])
    assert _attribute(0, _BX, _BY, low, np.array(["measured"]), players, _P, overhead_active=True)[0] == "A"


# ---- attribution margins (spec 8, 17.4) ----


def _two_team(vis=(True, True), icv=(pd.NA, pd.NA)) -> pd.DataFrame:
    # attributed A (team H) at 0.5 m; opponent B (team W) at 2.0 m.
    return _players(
        player_id=["A", "B"],
        x=[50.5, 52.0],
        y=[34.0, 34.0],
        team_id=["H", "W"],
        visibility=list(vis),
        in_camera_view=pd.array(list(icv), dtype="boolean"),
    )


def test_all_null_visibility_detection_aware_provider_raises_adr109():
    # A detection-aware provider (skillcorner) whose visibility is entirely null -> the pipeline discarded the
    # flag; _attribution_margins must RAISE (ADR-109 fail-loud), not silently read everyone as visible.
    players = _two_team(vis=(pd.NA, pd.NA))
    with pytest.raises(ValueError, match="detection flag"):
        _attribution_margins("A", 0.5, 0, _BX, _BY, players, footprint_active=False, provider="skillcorner")


def test_all_null_visibility_fully_observed_provider_assumes_observed():
    # A fully-observed provider (idsse) has no broadcast occlusion: all-null visibility -> assume observed
    # (every present player visible), no raise. The opponent at 2.0 m is visible, so the margin is finite.
    players = _two_team(vis=(pd.NA, pd.NA))
    m, _mp, ov, _r = _attribution_margins("A", 0.5, 0, _BX, _BY, players, footprint_active=False, provider="idsse")
    assert ov is True and math.isfinite(m)


def test_attribution_margin_is_other_team_minus_attributed():
    m, mp, ov, reason = _attribution_margins(
        "A", 0.5, 0, _BX, _BY, _two_team(), footprint_active=False, provider="skillcorner"
    )
    assert m == pytest.approx(1.5)  # d(B)=2.0 - d(A)=0.5
    assert mp == pytest.approx(1.5)  # nearest other player of any team is also B
    assert ov is True
    assert reason is pd.NA  # footprint gate inactive


def test_attribution_margin_nan_when_no_other_team_visible_never_inf():
    m, _mp, ov, _r = _attribution_margins(
        "A", 0.5, 0, _BX, _BY, _two_team(vis=(True, False)), footprint_active=False, provider="skillcorner"
    )
    assert math.isnan(m) and not math.isinf(m)
    assert ov is False


def test_unseen_reason_off_camera_vs_missed_detection_vs_na():
    # footprint active, opponent undetected: in_camera_view False -> off_camera; True -> missed_detection.
    _, _, _, r_off = _attribution_margins(
        "A",
        0.5,
        0,
        _BX,
        _BY,
        _two_team(vis=(True, False), icv=(pd.NA, False)),
        footprint_active=True,
        provider="skillcorner",
    )
    assert r_off == "off_camera"
    _, _, _, r_miss = _attribution_margins(
        "A",
        0.5,
        0,
        _BX,
        _BY,
        _two_team(vis=(True, False), icv=(pd.NA, True)),
        footprint_active=True,
        provider="skillcorner",
    )
    assert r_miss == "missed_detection"
    # visible opponent -> <NA>
    _, _, _, r_vis = _attribution_margins(
        "A", 0.5, 0, _BX, _BY, _two_team(), footprint_active=True, provider="skillcorner"
    )
    assert r_vis is pd.NA
    # footprint INACTIVE -> always <NA> (CL-16, fail-closed)
    _, _, _, r_off2 = _attribution_margins(
        "A",
        0.5,
        0,
        _BX,
        _BY,
        _two_team(vis=(True, False), icv=(pd.NA, False)),
        footprint_active=False,
        provider="skillcorner",
    )
    assert r_off2 is pd.NA


def test_no_provider_fails_closed_never_assumes_observed():
    # ADR-109 / C2 r2 C-06: with no data profile and no declared provider the visibility flag cannot be read; the
    # visibility-based columns are NaN / <NA> -- never a default "observed" and never read as skillcorner.
    for vis in ((True, True), (pd.NA, pd.NA)):
        m, mp, ov, r = _attribution_margins("A", 0.5, 0, _BX, _BY, _two_team(vis=vis), footprint_active=True)
        assert math.isnan(m) and math.isnan(mp) and ov is pd.NA and r is pd.NA


def test_detect_ball_contacts_provider_must_agree_with_the_profile():
    from silly_kicks.tracking._restart_validation import DataProfile

    frames = pd.DataFrame(
        {
            "is_ball": [True],
            "period_id": [1],
            "frame_id": [1],
            "time_seconds": [0.0],
            "x": [np.nan],
            "y": [np.nan],
            "player_id": [None],
        }
    )
    prof = DataProfile("skillcorner", "r1", "g1", "2025")
    with pytest.raises(ValueError, match="disagrees"):
        detect_ball_contacts(frames, data_profile=prof, provider="idsse")
    detect_ball_contacts(frames, data_profile=prof, provider="skillcorner")  # agreeing: no raise


def test_detect_ball_contacts_has_the_attribution_columns():
    # An all-ball-undetected frame -> empty contacts, but the schema carries the four new columns.
    frames = pd.DataFrame(
        {
            "is_ball": [True],
            "period_id": [1],
            "frame_id": [1],
            "time_seconds": [0.0],
            "x": [np.nan],
            "y": [np.nan],
            "player_id": [None],
        }
    )
    out = detect_ball_contacts(frames)
    for c in (
        "attribution_margin",
        "attribution_margin_player",
        "attribution_other_visible",
        "attribution_other_unseen_reason",
    ):
        assert c in out.columns
