"""Task 6: pass-risk calibration consumer (`scripts/validate_pass_risk_calibration.py`)."""

from __future__ import annotations

import pandas as pd


def test_headline_is_completed_only_false_alarm_rate():
    from scripts.validate_pass_risk_calibration import compute_pass_risk_metrics

    df = pd.DataFrame({"control": [0.05, 0.15, 0.5, 0.05], "is_completed": [True, True, True, False]})
    m = compute_pass_risk_metrics(df)
    # P(control < 0.1 | completed): completed = idx 0,1,2 -> control<0.1 at idx0 -> 1/3
    assert abs(m["headline_false_alarm_rate"]["0.1"] - 1 / 3) < 1e-9
    assert m["low_control_completion_band"]["contaminated"] is True
    assert "OVER-PREDICTION" in m["scope_note"]
