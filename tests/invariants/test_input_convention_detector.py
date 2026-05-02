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
