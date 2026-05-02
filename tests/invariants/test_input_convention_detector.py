"""End-to-end input-convention detector validation against real fixtures.

The unit tests for ``detect_input_convention`` live in
``tests/spadl/test_orientation.py`` and use synthetic data. This file is the
counterpart that exercises the detector against the same real-shape provider
fixtures used by the rest of the invariant suite -- closing the loop between
synthetic correctness and real-data correctness.

Each test runs the detector against the RAW pre-conversion events for a
provider and asserts the detector returns the convention each provider's
docstring declares. If silly-kicks ever forks an upstream loader change that
silently shifts the input convention, this test catches it before any silent
data corruption.

The Opta synthetic fixture in ``_loaders.load_opta_2team_synthetic`` is in
absolute_no_switch convention -- the convention silly-kicks's Opta converter
expects per its docstring contract (PR-S22 / ADR-006).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl.orientation import (
    ABSOLUTE_FRAME_HOME_RIGHT,
    PER_PERIOD_ABSOLUTE,
    POSSESSION_PERSPECTIVE,
    InputConvention,
    detect_input_convention,
    validate_input_convention,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Real-fixture builders for the detector (raw pre-conversion shape, NOT SPADL)
# ---------------------------------------------------------------------------


def _build_statsbomb_raw_events_for_detector(match_id: int) -> pd.DataFrame:
    """Build per-event rows with team_id, period_id, start_x, is_shot for the StatsBomb fixture."""
    fixture_path = _REPO_ROOT / "tests" / "datasets" / "statsbomb" / "raw" / "events" / f"{match_id}.json"
    with open(fixture_path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for e in raw:
        loc = e.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        team_id = (e.get("team") or {}).get("id")
        if team_id is None:
            continue
        rows.append(
            {
                "game_id": match_id,
                "team_id": team_id,
                "period_id": e.get("period"),
                "start_x": float(loc[0]),
                "is_shot": (e.get("type") or {}).get("name") == "Shot",
            }
        )
    return pd.DataFrame(rows)


def _build_idsse_raw_events_for_detector() -> pd.DataFrame:
    """Build the per-row detector input from the IDSSE bronze parquet."""
    parquet_path = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
    events = pd.read_parquet(parquet_path)
    df = pd.DataFrame(
        {
            "game_id": 1,
            "team_id": events["team"],
            "period_id": events["period"] if "period" in events.columns else 1,
            "start_x": events["x_source_position"],
            "is_shot": events["event_type"] == "ShotAtGoal",
        }
    )
    df = df.dropna(subset=["team_id", "period_id", "start_x"])
    df = df[df["team_id"] != "unknown"]
    return df


def _build_opta_synthetic_raw_events_for_detector() -> pd.DataFrame:
    """Synthetic Opta absolute_no_switch event set in raw 0-100 frame."""
    HOME, AWAY = 100, 200
    rows = []
    for period in (1, 2):
        for shot_num in range(12):
            rows.append(
                {"game_id": 1, "team_id": HOME, "period_id": period, "start_x": 80.0 + shot_num, "is_shot": True}
            )
            rows.append(
                {"game_id": 1, "team_id": AWAY, "period_id": period, "start_x": 8.0 + shot_num, "is_shot": True}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection tests against real fixtures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("match_id", [7298, 7584, 3754058])
def test_statsbomb_raw_detected_as_possession_perspective(match_id: int):
    events = _build_statsbomb_raw_events_for_detector(match_id)
    result = detect_input_convention(events, match_col="game_id", x_max=120.0, is_shot_col="is_shot")
    assert result.convention is InputConvention.POSSESSION_PERSPECTIVE, result.diagnostics
    assert result.confidence in ("high", "medium")


def test_idsse_raw_detected_as_absolute_frame_home_right():
    events = _build_idsse_raw_events_for_detector()
    result = detect_input_convention(events, match_col="game_id", x_max=105.0, is_shot_col="is_shot")
    # IDSSE bronze fixture has only 2 shots total -- below the medium threshold per
    # (team, period). Detector should defer (None) rather than misclassify.
    assert result.convention is None
    assert result.confidence in ("low", "ambiguous")


def test_opta_synthetic_detected_as_absolute_frame_home_right():
    events = _build_opta_synthetic_raw_events_for_detector()
    result = detect_input_convention(events, match_col="game_id", x_max=100.0, is_shot_col="is_shot")
    assert result.convention is InputConvention.ABSOLUTE_FRAME_HOME_RIGHT
    assert result.confidence == "high"


# ---------------------------------------------------------------------------
# Validator semantics — full chain (real fixture + declared mismatch)
# ---------------------------------------------------------------------------


def test_validator_silent_when_statsbomb_declared_correctly():
    events = _build_statsbomb_raw_events_for_detector(7298)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_input_convention(
            events,
            declared=POSSESSION_PERSPECTIVE,
            match_col="game_id",
            x_max=120.0,
            is_shot_col="is_shot",
            on_mismatch="warn",
        )


def test_validator_warns_when_statsbomb_declared_as_absolute_frame():
    events = _build_statsbomb_raw_events_for_detector(7298)
    with pytest.warns(UserWarning, match="declared=absolute_frame_home_right"):
        validate_input_convention(
            events,
            declared=ABSOLUTE_FRAME_HOME_RIGHT,
            match_col="game_id",
            x_max=120.0,
            is_shot_col="is_shot",
            on_mismatch="warn",
        )


def test_validator_silent_on_idsse_because_signal_too_weak():
    """IDSSE bronze has only 2 shots -- detector returns None, validator defers silently."""
    events = _build_idsse_raw_events_for_detector()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_input_convention(
            events,
            declared=ABSOLUTE_FRAME_HOME_RIGHT,
            match_col="game_id",
            x_max=105.0,
            is_shot_col="is_shot",
            on_mismatch="warn",
        )


# ---------------------------------------------------------------------------
# TF-22 (PR-S23 / silly-kicks 3.0.1): detector must NOT false-positive
# ABSOLUTE_FRAME_HOME_RIGHT on sparse per-team-period-asymmetric per-period-
# absolute data. The IDSSE J03WMX fixture (4/5/8/3 shots distribution) is
# the canonical empirical example -- see lakehouse SK3-MIG bug report.
# ---------------------------------------------------------------------------


def _build_synthetic_per_period_events(
    *,
    p1_home_x_mean: float,
    p1_home_n: int,
    p1_away_x_mean: float,
    p1_away_n: int,
    p2_home_x_mean: float,
    p2_home_n: int,
    p2_away_x_mean: float,
    p2_away_n: int,
    x_max: float = 105.0,
) -> pd.DataFrame:
    """Synthetic 1-match events with controlled per-period-per-team shot counts + means.

    Shot x values are tightly clustered around each requested mean so the
    detector's >mid_x classification is unambiguous.
    """
    import numpy as np

    rng = np.random.default_rng(seed=42)
    rows: list[dict[str, object]] = []
    eid = 0
    for period, side, n, x_mean in (
        (1, "home", p1_home_n, p1_home_x_mean),
        (1, "away", p1_away_n, p1_away_x_mean),
        (2, "home", p2_home_n, p2_home_x_mean),
        (2, "away", p2_away_n, p2_away_x_mean),
    ):
        for _ in range(n):
            eid += 1
            x = float(np.clip(rng.normal(loc=x_mean, scale=0.5), 0.0, x_max))
            rows.append(
                {
                    "match_id": "synth-001",
                    "team": side,
                    "period": period,
                    "x": x,
                    "is_shot": True,
                }
            )
    return pd.DataFrame(rows)


def test_tf22_detector_sparse_per_team_period_asymmetric_returns_low_confidence_none():
    """IDSSE J03WMX shape: home reliable in P1 only, away reliable in P2 only.

    Each team has reliable data in 1 distinct period -- detector cannot
    distinguish ABSOLUTE_FRAME_HOME_RIGHT from PER_PERIOD_ABSOLUTE on
    shot-only signal. Must return convention=None, confidence='low'
    (post-TF-22 hardening; before TF-22 this false-positives ABSOLUTE).
    """
    events = _build_synthetic_per_period_events(
        p1_home_x_mean=13.0,
        p1_home_n=5,
        p1_away_x_mean=11.0,
        p1_away_n=4,  # below medium threshold (n<5)
        p2_home_x_mean=92.0,
        p2_home_n=3,  # below medium threshold (n<5)
        p2_away_x_mean=95.0,
        p2_away_n=8,
    )
    result = detect_input_convention(
        events,
        match_col="match_id",
        x_max=105.0,
        x_col="x",
        team_col="team",
        period_col="period",
        is_shot_col="is_shot",
    )
    assert result.convention is None, (
        f"expected convention=None for sparse per-team-period-asymmetric data, "
        f"got {result.convention!r}; diagnostics={result.diagnostics}"
    )
    assert result.confidence == "low", f"expected confidence='low', got {result.confidence!r}"


def test_tf22_detector_dense_per_period_absolute_classifies_correctly():
    """Dense per-period-absolute: each team has >=5 shots in EACH period.

    Both teams alternate sides between periods -- must classify
    PER_PERIOD_ABSOLUTE with confidence='high'.
    """
    events = _build_synthetic_per_period_events(
        p1_home_x_mean=15.0,
        p1_home_n=10,
        p1_away_x_mean=90.0,
        p1_away_n=10,
        p2_home_x_mean=92.0,
        p2_home_n=10,
        p2_away_x_mean=13.0,
        p2_away_n=10,
    )
    result = detect_input_convention(
        events,
        match_col="match_id",
        x_max=105.0,
        x_col="x",
        team_col="team",
        period_col="period",
        is_shot_col="is_shot",
    )
    assert result.convention is PER_PERIOD_ABSOLUTE, (
        f"expected PER_PERIOD_ABSOLUTE, got {result.convention!r}; diagnostics={result.diagnostics}"
    )
    assert result.confidence == "high"


def test_tf22_detector_dense_absolute_frame_home_right_classifies_correctly():
    """Dense absolute-frame-home-right: each team consistently on one side across both periods.

    Sanity check that TF-22 hardening didn't regress dense-data classification.
    """
    events = _build_synthetic_per_period_events(
        p1_home_x_mean=92.0,
        p1_home_n=10,
        p1_away_x_mean=13.0,
        p1_away_n=10,
        p2_home_x_mean=92.0,
        p2_home_n=10,
        p2_away_x_mean=13.0,
        p2_away_n=10,
    )
    result = detect_input_convention(
        events,
        match_col="match_id",
        x_max=105.0,
        x_col="x",
        team_col="team",
        period_col="period",
        is_shot_col="is_shot",
    )
    assert result.convention is ABSOLUTE_FRAME_HOME_RIGHT, (
        f"expected ABSOLUTE_FRAME_HOME_RIGHT, got {result.convention!r}; diagnostics={result.diagnostics}"
    )
    assert result.confidence == "high"
