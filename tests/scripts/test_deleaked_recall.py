"""Task 13: recall honesty once the failed-pass target is model-inferred."""

from __future__ import annotations

import pandas as pd

from scripts.validate_cover_shadow_rq1 import compute_cover_shadow_metrics

_BASE = {
    "is_blocked_majority": [True, False, True],
    "is_completed": [False, True, False],
    "is_fail": [True, False, True],
    "is_cross": [False, False, False],
    "p_blocked_center": [0.9, 0.1, 0.8],
    "p_blocked_mean": [0.8, 0.1, 0.7],
    "p_blocked_max": [0.95, 0.2, 0.85],
    "p_received_center": [0.3, 0.6, 0.3],
    "p_received_left": [0.3, 0.6, 0.3],
    "p_received_right": [0.3, 0.6, 0.3],
    "n_blocked": [3, 0, 2],
}


def test_deleaked_block_flags_deleak_and_carries_caveats():
    df = pd.DataFrame({**_BASE, "target_source": ["intended_receiver", "receiver", "intended_receiver"]})
    d = compute_cover_shadow_metrics(df)["deleaked_recall"]
    assert d["is_deleaked"] is True
    assert d["target_source_counts"]["intended_receiver"] == 2
    assert "UPPER BOUND" in d["failed_pass_validity_note"]  # R1
    assert "ROBUSTNESS" in d["robustness_band_note"]  # M3
    assert "UNMEASURED" in d["residual_bias_note"]  # R3
    assert "candidate-count" in d["candidate_count_shift_note"]  # M2


def test_leaked_build_is_not_flagged_deleaked():
    df = pd.DataFrame({**_BASE, "target_source": ["end_xy", "receiver", "end_xy"]})
    d = compute_cover_shadow_metrics(df)["deleaked_recall"]
    assert d["is_deleaked"] is False  # 4.87.0-style leaked build
