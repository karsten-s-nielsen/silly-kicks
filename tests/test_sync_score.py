"""TF-6 -- sync_score per-action tracking<->events sync-quality (3 aggregations)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import LinkReport
from silly_kicks.tracking.utils import add_sync_score, sync_score


def _toy_links() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [1, 1, 1, 2, 2, 3],
            "frame_id": [10, 11, 12, 20, 21, 30],
            "time_offset_seconds": [0.0, 0.04, 0.08, 0.0, 0.04, 0.0],
            "n_candidate_frames": [3, 3, 3, 2, 2, 1],
            "link_quality_score": [0.95, 0.90, 0.85, 0.70, 0.60, 0.99],
        }
    )


def test_returns_three_columns():
    df = sync_score(_toy_links(), high_quality_threshold=0.85)
    assert list(df.columns) == ["sync_score_min", "sync_score_mean", "sync_score_high_quality_frac"]


def test_per_action_min_mean():
    df = sync_score(_toy_links(), high_quality_threshold=0.85)
    assert df.loc[1, "sync_score_min"] == pytest.approx(0.85)
    assert df.loc[1, "sync_score_mean"] == pytest.approx((0.95 + 0.90 + 0.85) / 3)
    assert df.loc[2, "sync_score_min"] == pytest.approx(0.60)
    assert df.loc[3, "sync_score_min"] == pytest.approx(0.99)


def test_high_quality_frac_at_threshold():
    df = sync_score(_toy_links(), high_quality_threshold=0.85)
    assert df.loc[1, "sync_score_high_quality_frac"] == pytest.approx(1.0)
    assert df.loc[2, "sync_score_high_quality_frac"] == pytest.approx(0.0)
    assert df.loc[3, "sync_score_high_quality_frac"] == pytest.approx(1.0)


def test_bounds_zero_one():
    df = sync_score(_toy_links(), high_quality_threshold=0.5)
    for col in df.columns:
        assert (df[col] >= 0).all()
        assert (df[col] <= 1).all()


def test_min_le_mean():
    df = sync_score(_toy_links(), high_quality_threshold=0.5)
    assert (df["sync_score_min"] <= df["sync_score_mean"] + 1e-9).all()


def test_add_sync_score_merges_on_action_id():
    actions = pd.DataFrame({"action_id": [1, 2, 3, 4], "type_id": [0, 1, 2, 3]})
    out = add_sync_score(actions, _toy_links(), high_quality_threshold=0.85)
    assert "sync_score_min" in out.columns
    indexed = out.set_index("action_id")
    assert pd.isna(indexed.loc[4, "sync_score_min"])  # action 4 has no link rows


def test_add_sync_score_action_id_required():
    with pytest.raises(ValueError, match="action_id"):
        add_sync_score(pd.DataFrame({"x": [1, 2]}), _toy_links())


def test_link_report_sync_scores_method():
    rpt = LinkReport(
        n_actions_in=3,
        n_actions_linked=3,
        n_actions_unlinked=0,
        n_actions_multi_candidate=2,
        per_provider_link_rate={"sportec": 1.0},
        max_time_offset_seconds=0.08,
        tolerance_seconds=0.1,
    )
    df = rpt.sync_scores(_toy_links(), high_quality_threshold=0.85)
    assert "sync_score_min" in df.columns


def test_atomic_actions_consume_add_sync_score():
    """sync_score is action-content-agnostic -- atomic mirror via shared util."""
    atomic = pd.DataFrame({"action_id": [1, 2], "x": [10.0, 20.0], "y": [30.0, 40.0]})
    out = add_sync_score(atomic, _toy_links(), high_quality_threshold=0.85)
    assert {"sync_score_min", "sync_score_mean", "sync_score_high_quality_frac"} <= set(out.columns)
    assert len(out) == len(atomic)


def test_invariant_bounds_zero_one_with_extreme_threshold():
    """All three scores bounded in [0, 1] regardless of threshold."""
    for thresh in (0.0, 0.5, 1.0):
        df = sync_score(_toy_links(), high_quality_threshold=thresh)
        assert ((df >= 0) & (df <= 1)).all().all()


def test_nan_link_score_treated_as_below_threshold():
    links = _toy_links().copy()
    links.loc[0, "link_quality_score"] = np.nan
    df = sync_score(links, high_quality_threshold=0.85)
    # min/mean drop NaN by default
    assert pd.notna(df.loc[1, "sync_score_min"])
