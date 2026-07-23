"""Unit tests for the shared `_blocked_flag` nullable-boolean helper (TF-51 prereq)."""

import numpy as np
import pandas as pd

from silly_kicks.spadl.utils import _blocked_flag


def test_all_na_when_not_applicable():
    col = _blocked_flag(3)
    assert str(col.dtype) == "boolean"
    assert col.isna().all()


def test_true_false_on_applicable_na_elsewhere():
    # rows 0,2 are shots (applicable); row 0 blocked, row 2 not; row 1 is a non-shot.
    applicable = np.array([True, False, True])
    blocked = np.array([True, False, False])
    col = _blocked_flag(3, applicable=applicable, blocked=blocked)
    assert str(col.dtype) == "boolean"
    assert col[0] == True  # noqa: E712  blocked shot
    assert pd.isna(col[1])  # non-shot -> NA, never False
    assert col[2] == False  # noqa: E712  shot, not blocked


def test_nan_in_blocked_coerced_to_false_not_true():
    # astype(bool) would turn NA -> True; the helper must coerce NA -> False.
    applicable = np.array([True, True])
    blocked = pd.array([True, pd.NA], dtype="boolean")  # a NA-bearing blocked signal
    col = _blocked_flag(2, applicable=applicable, blocked=blocked)
    assert col[0] == True  # noqa: E712
    assert col[1] == False  # noqa: E712  NA -> False, NOT True
