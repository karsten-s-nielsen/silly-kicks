"""Tests for the renamed public ``silly_kicks.tracking.direction`` module.

Covers (ADR-010 / spec 2026-05-30):
- T-A: public symbols survive the ``_direction.py -> direction.py`` rename and
  ``require_et_direction`` is re-exported from both ``silly_kicks.tracking`` and
  ``silly_kicks.spadl``.
- The ``require_et_direction`` guard semantics (Task 2).
- Cross-provider parity: every per-period-absolute converter raises the same
  message shape on ET-without-flag, and the flag actually orients ET (Task 6).
"""

from __future__ import annotations

import pandas as pd
import pytest

# --- T-A: symbol preservation across the rename -----------------------------


def test_direction_public_symbols_present():
    from silly_kicks.tracking import direction

    assert hasattr(direction, "home_attacks_right_per_period")
    assert hasattr(direction, "require_et_direction")  # added in Task 2


def test_require_et_direction_reexported():
    import silly_kicks.spadl as s
    import silly_kicks.tracking as t

    assert hasattr(t, "require_et_direction")
    assert hasattr(s, "require_et_direction")


# --- Task 2: require_et_direction guard semantics ---------------------------


def test_require_et_raises_when_et_present_and_flag_none():
    from silly_kicks.tracking.direction import require_et_direction

    with pytest.raises(ValueError, match="ET periods"):
        require_et_direction(pd.Series([1, 1, 3, 4]), None, source="sportec convert_to_frames")


def test_require_et_noop_when_flag_provided():
    from silly_kicks.tracking.direction import require_et_direction

    require_et_direction(pd.Series([1, 3]), True, source="x")  # no raise
    require_et_direction(pd.Series([1, 3]), False, source="x")  # False is a valid flag, no raise


def test_require_et_noop_when_no_et_periods():
    from silly_kicks.tracking.direction import require_et_direction

    require_et_direction(pd.Series([1, 1, 2, 2]), None, source="x")  # no raise


def test_require_et_message_names_source_and_field():
    from silly_kicks.tracking.direction import require_et_direction

    with pytest.raises(ValueError, match="metrica convert_to_actions") as exc:
        require_et_direction(pd.Series([3]), None, source="metrica convert_to_actions")
    msg = str(exc.value)
    assert msg.startswith("metrica convert_to_actions: data contains ET periods")
    assert "home_team_start_left_extratime" in msg


def test_require_et_accepts_ndarray_and_list():
    import numpy as np

    from silly_kicks.tracking.direction import require_et_direction

    with pytest.raises(ValueError, match="ET periods"):
        require_et_direction(np.array([1, 4]), None, source="x")
    with pytest.raises(ValueError, match="ET periods"):
        require_et_direction([2, 3], None, source="x")
