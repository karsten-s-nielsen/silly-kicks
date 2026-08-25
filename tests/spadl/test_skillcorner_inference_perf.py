"""Structural perf guard for infer_defensive_actions (ADR-068): the direct_regain OBE candidate
lookup is pre-grouped ONCE, not re-filtered over the whole obe table per defensive row."""

import pandas as pd

import silly_kicks.spadl._skillcorner_inference as _sci
from silly_kicks.spadl._skillcorner_inference import infer_defensive_actions
from tests._perf_structural import call_counter


def test_obe_lookup_built_once(monkeypatch):
    calls = call_counter(monkeypatch, _sci, "group_rows")
    # Two defensive-start rows -> the OBE candidate scan runs twice; group_rows must be built ONCE.
    pp = pd.DataFrame(
        {
            "event_id": ["pp_1", "pp_2"],
            "period": [1, 1],
            "time_seconds": [10.0, 15.0],
            "team_id": ["team_b", "team_b"],
            "player_id": ["p11", "p12"],
            "start_type": ["pass_interception", "recovery"],
            "x_start": [5.0, 15.0],
            "y_start": [3.0, 10.0],
        }
    )
    obe = pd.DataFrame(
        {
            "period": [1, 1],
            "time_seconds": [9.9, 14.8],
            "team_id": ["team_b", "team_b"],
            "player_id": ["p13", "p14"],
            "end_type": ["direct_regain", "direct_regain"],
            "x_start": [4.0, 14.0],
            "y_start": [2.0, 9.0],
        }
    )
    result = infer_defensive_actions(pp, obe)
    assert calls["n"] == 1  # once total; pre-ADR-068 the full obe_regains table was filtered per row
    # both defensive rows upgraded to tackle via their nearest direct_regain (behaviour preserved)
    assert len(result) == 2
    assert set(result["player_id"]) == {"p13", "p14"}
