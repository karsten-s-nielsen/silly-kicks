"""Tests for the shared training-exclusion helper (TF-46 Arm 1, decision B5/E)."""

import pandas as pd

from silly_kicks.spadl._training_exclusion import training_exclusion_mask


def _frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {"action_id": list(range(n))}
    base.update(cols)
    return pd.DataFrame(base)


def test_excludes_inferred_and_stopgap_result_source():
    df = _frame(result_source=["native", "inferred", "stopgap", pd.NA])
    mask = training_exclusion_mask(df)
    assert list(mask) == [False, True, True, False]


def test_excludes_chain_ambiguous_restart_evidence():
    df = _frame(restart_evidence=["tagged", "chain", "chain_ambiguous", pd.NA])
    mask = training_exclusion_mask(df)
    assert list(mask) == [False, False, True, False]


def test_excludes_any_repair_kind_row():
    df = _frame(repair_kind=[pd.NA, "owngoal", "missing_goal", "own_goal_unsignalled"])
    mask = training_exclusion_mask(df)
    assert list(mask) == [False, True, True, True]


def test_keeps_na_and_native():
    df = _frame(
        result_source=["native", pd.NA],
        restart_evidence=[pd.NA, "tagged"],
        repair_kind=[pd.NA, pd.NA],
    )
    assert list(training_exclusion_mask(df)) == [False, False]


def test_combined_columns_union():
    df = _frame(
        result_source=["native", "inferred", "native"],
        restart_evidence=["tagged", "tagged", "chain_ambiguous"],
        repair_kind=[pd.NA, pd.NA, pd.NA],
    )
    assert list(training_exclusion_mask(df)) == [False, True, True]


def test_tolerates_all_columns_missing_hand_built_frame():
    # A hand-built frame with none of the signal columns -> all-False, no raise (decision E).
    df = _frame(type_id=[0, 1, 2])
    mask = training_exclusion_mask(df)
    assert list(mask) == [False, False, False]


def test_tolerates_missing_columns_on_atomic_like_frame():
    # Atomic SPADL drops the signal columns; the helper must not raise on such a frame.
    df = _frame(x=[1.0, 2.0], dx=[0.1, 0.2])
    mask = training_exclusion_mask(df)
    assert list(mask) == [False, False]


def test_mask_is_index_aligned():
    df = _frame(result_source=["inferred", "native"])
    df.index = [10, 20]
    mask = training_exclusion_mask(df)
    assert list(mask.index) == [10, 20]
