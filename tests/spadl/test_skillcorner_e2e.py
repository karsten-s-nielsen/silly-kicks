"""SkillCorner SPADL converter e2e tests.

Requires the 10 A-League matches downloaded from pining-for-the-data API
at C:\\Users\\Karsten\\AppData\\Local\\Temp\\skillcorner_sample\\
"""

import json

import pandas as pd
import pytest

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import SKILLCORNER_SPADL_COLUMNS
from silly_kicks.spadl.skillcorner import convert_to_actions
from tests._skillcorner_sample import MATCH_IDS as _MATCH_IDS
from tests._skillcorner_sample import SAMPLE_DIR as _SAMPLE_DIR
from tests._skillcorner_sample import available_matches
from tests._skillcorner_sample import find_artifact as _find_artifact


def _has_data():
    return bool(available_matches(("_dynamic_events.csv", "_match.json")))


pytestmark = pytest.mark.e2e


@pytest.fixture(params=_MATCH_IDS)
def match_data(request):
    match_dir = _SAMPLE_DIR / request.param
    events_path = _find_artifact(match_dir, "dynamic_events.csv")
    meta_path = _find_artifact(match_dir, "match.json")
    if events_path is None or meta_path is None:
        pytest.skip(f"Match {request.param} not available at {match_dir}")
    events = pd.read_csv(events_path, low_memory=False)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return events, meta, request.param


@pytest.mark.skipif(not _has_data(), reason="SkillCorner sample data not available")
class TestSkillcornerE2E:
    def test_no_crash(self, match_data):
        events, meta, _mid = match_data
        actions, _report = convert_to_actions(events, meta)
        assert len(actions) > 0

    def test_schema_matches(self, match_data):
        events, meta, _mid = match_data
        actions, _ = convert_to_actions(events, meta)
        assert set(actions.columns) == set(SKILLCORNER_SPADL_COLUMNS.keys())

    def test_no_unrecognized_events(self, match_data):
        events, meta, mid = match_data
        _, report = convert_to_actions(events, meta)
        assert not report.has_unrecognized, f"Match {mid}: {report.unrecognized_counts}"

    def test_monotonic_time_per_period(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        for period in actions["period_id"].unique():
            times = actions[actions["period_id"] == period]["time_seconds"].values
            assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), (
                f"Match {mid}: non-monotonic in period {period}"
            )

    def test_time_is_period_relative_not_continuous_clock(self, match_data):
        # BUG 1 regression guard: time_seconds must be PERIOD-RELATIVE (ADR-017), so each
        # period's first action starts near 0 -- NOT the continuous broadcast clock (which put
        # 2nd-half actions at 2700+ s and broke action<->frame linkage). Monotonicity alone
        # (above) passes on a continuous clock too, which is why this slipped through.
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        for period in actions["period_id"].unique():
            first = actions[actions["period_id"] == period]["time_seconds"].min()
            assert first < 600.0, (
                f"Match {mid}: period {period} first action at {first:.0f}s -- looks like a "
                "continuous match clock, not period-relative (BUG 1)."
            )

    def test_has_interceptions(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        interceptions = actions[actions["type_id"] == spadlconfig.actiontype_id["interception"]]
        # Every match should have some interceptions (from start_type)
        assert len(interceptions) > 0, f"Match {mid}: no interceptions"

    def test_coordinates_in_range(self, match_data):
        events, meta, mid = match_data
        actions, _ = convert_to_actions(events, meta)
        for col in ["start_x", "end_x"]:
            valid = actions[col].dropna()
            assert (valid >= 0.0).all(), f"Match {mid}: {col} below 0"
            assert (valid <= 105.0).all(), f"Match {mid}: {col} above 105"
        for col in ["start_y", "end_y"]:
            valid = actions[col].dropna()
            assert (valid >= 0.0).all(), f"Match {mid}: {col} below 0"
            assert (valid <= 68.0).all(), f"Match {mid}: {col} above 68"
