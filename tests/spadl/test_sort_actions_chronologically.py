"""Unit tests for the shared chronological-sort seam (Task 1).

``sort_actions_chronologically`` is the one stable-sort primitive every SPADL
converter uses at the top of its frame, before any positional (``.shift()``)
derivation.  It must sort by the ordering key, preserve genuine ties (stable),
never mutate its input, and -- crucially -- RAISE on an absent ordering key
rather than silently degrade to a partial sort.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.spadl.base import sort_actions_chronologically


def test_stable_sort_orders_by_key_and_preserves_ties():
    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "time_seconds": [2.0, 1.0, 1.0],
            "tag": ["late", "tieA", "tieB"],
        }
    )
    out = sort_actions_chronologically(df)
    assert list(out["tag"]) == ["tieA", "tieB", "late"]  # time order; ties keep input order (stable)


def test_custom_by_columns_for_raw_event_frames():
    df = pd.DataFrame({"period_id": [1, 1], "milliseconds": [500, 100], "tag": ["b", "a"]})
    out = sort_actions_chronologically(df, by=("period_id", "milliseconds"))
    assert list(out["tag"]) == ["a", "b"]


def test_tiebreak_breaks_equal_time_deterministically():
    df = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [1.0, 1.0],
            "__order__": [1.5, 1.0],
            "tag": ["synth", "parent"],
        }
    )
    out = sort_actions_chronologically(df, tiebreak=("__order__",))
    assert list(out["tag"]) == ["parent", "synth"]


def test_empty_frame_passes_through():
    df = pd.DataFrame({"game_id": [], "period_id": [], "time_seconds": []})
    assert len(sort_actions_chronologically(df)) == 0


def test_does_not_mutate_input():
    df = pd.DataFrame({"game_id": [1, 1], "period_id": [1, 1], "time_seconds": [2.0, 1.0]})
    before = df.copy()
    sort_actions_chronologically(df)
    pd.testing.assert_frame_equal(df, before)


def test_missing_ordering_key_raises_not_silently_partial_sorts():  # M-D
    df = pd.DataFrame({"period_id": [1, 1], "tag": ["b", "a"]})  # no time_seconds
    with pytest.raises(KeyError, match=r"absent|ordering key"):
        sort_actions_chronologically(df)  # default by includes time_seconds
    with pytest.raises(KeyError):
        sort_actions_chronologically(df, by=("period_id", "milliseconds"))  # mistyped/absent time col
