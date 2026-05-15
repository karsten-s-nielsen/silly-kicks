"""Physical invariants for SkillCorner SPADL converter.

Verify that converted actions satisfy spatial properties that must hold
regardless of the specific match data.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.spadl.skillcorner import convert_to_actions

_FIXTURE_DIR = Path(__file__).parent.parent / "datasets" / "skillcorner"


@pytest.fixture
def actions():
    events = pd.read_csv(_FIXTURE_DIR / "basic_possessions.csv")
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        meta = json.load(f)
    actions, _ = convert_to_actions(events, meta)
    return actions


class TestSpatialInvariants:
    """Coordinates must be in valid SPADL range."""

    def test_start_x_in_range(self, actions):
        valid = actions["start_x"].dropna()
        assert (valid >= 0.0).all(), f"start_x below 0: {valid[valid < 0.0].tolist()}"
        assert (valid <= 105.0).all(), f"start_x above 105: {valid[valid > 105.0].tolist()}"

    def test_start_y_in_range(self, actions):
        valid = actions["start_y"].dropna()
        assert (valid >= 0.0).all()
        assert (valid <= 68.0).all()

    def test_end_x_in_range(self, actions):
        valid = actions["end_x"].dropna()
        assert (valid >= 0.0).all(), f"end_x below 0: {valid[valid < 0.0].tolist()}"
        assert (valid <= 105.0).all(), f"end_x above 105: {valid[valid > 105.0].tolist()}"

    def test_end_y_in_range(self, actions):
        valid = actions["end_y"].dropna()
        assert (valid >= 0.0).all(), f"end_y below 0: {valid[valid < 0.0].tolist()}"
        assert (valid <= 68.0).all(), f"end_y above 68: {valid[valid > 68.0].tolist()}"


class TestTemporalInvariants:
    """Time must be monotonic within periods."""

    def test_time_seconds_non_negative(self, actions):
        assert (actions["time_seconds"] >= 0).all()

    def test_time_seconds_monotonic_within_period(self, actions):
        for period in actions["period_id"].unique():
            period_actions = actions[actions["period_id"] == period]
            times = period_actions["time_seconds"].values
            # Allow equal timestamps but not decreasing
            assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), (
                f"Non-monotonic time_seconds in period {period}"
            )
