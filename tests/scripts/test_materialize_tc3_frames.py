"""The materialized frames must match what the trainer's established input contains.

The trainer's existing input comes from a different pipeline. If the pining parse yields a different
schema, dtype set, or row filtering, the trainer silently fits on DIFFERENT DATA -- the same
train/serve-skew family this cycle exists to close, arriving through the fix for it, and landing
UNDERNEATH the D2 measurement built to detect trouble.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.materialize_tc3_frames import assert_frames_parity


def _frame(**over) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "frame_id": [1, 2],
            "player_id": ["a", "b"],
            "x": [1.0, 2.0],
            "y": [3.0, 4.0],
            "team_id": ["H", "A"],
            "vx": [0.5, 0.6],
        }
    )
    for k, v in over.items():
        base[k] = v
    return base


def test_identical_frames_pass():
    assert_frames_parity(_frame(), _frame(), match_id="m1")


def test_missing_column_is_rejected():
    with pytest.raises(AssertionError, match="column"):
        assert_frames_parity(_frame().drop(columns=["team_id"]), _frame(), match_id="m1")


def test_row_count_mismatch_is_rejected():
    with pytest.raises(AssertionError, match="row count"):
        assert_frames_parity(_frame().iloc[:1], _frame(), match_id="m1")


def test_dtype_drift_is_rejected():
    with pytest.raises(AssertionError, match="dtype"):
        assert_frames_parity(_frame(x=[1, 2]), _frame(), match_id="m1")


def test_value_drift_is_rejected():
    """Compare CONTENT, not just shape -- a schema-equal frame with different coordinates is
    exactly the silent-skew case."""
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(_frame(x=[1.5, 2.0]), _frame(), match_id="m1")


def test_NON_KEY_value_drift_is_also_rejected():
    """Measured defect in the first draft: hashing only the identity columns let `vx` drift from
    0.5 to 99.0 undetected. Ghost's extractor consumes velocity and so does `infer_ball_carrier`,
    so a positions-right / velocities-wrong parse is precisely the silent skew this gate names."""
    with pytest.raises(AssertionError, match="checksum"):
        assert_frames_parity(_frame(vx=[99.0, 0.6]), _frame(), match_id="m1")


def test_negative_zero_does_not_trip_a_spurious_failure():
    """`-0.0` and `0.0` hash differently unless normalised. Negative zero is reachable via the
    velocity NEGATION (`-vx` where `vx == 0.0`), and the corpus driver treats a parity failure as
    STOP -- so this would cost a corpus pass to diagnose."""
    assert_frames_parity(_frame(x=[-0.0, 2.0]), _frame(x=[0.0, 2.0]), match_id="m1")


def test_duplicate_identity_rows_are_order_insensitive():
    """Two rows tying on every identity column but differing on `vx` must hash the same regardless
    of order. They did not, until the sort key was widened to ALL columns -- and GS duplicate frames
    make this reachable, at the same spurious-STOP cost as negative zero."""
    dup = _frame()
    dup["frame_id"] = [1, 1]
    dup["player_id"] = ["a", "a"]
    dup["x"] = [1.0, 1.0]
    dup["y"] = [3.0, 3.0]
    dup["team_id"] = ["H", "H"]
    dup["vx"] = [0.5, 9.9]
    assert_frames_parity(dup, dup.iloc[::-1].reset_index(drop=True), match_id="m1")
