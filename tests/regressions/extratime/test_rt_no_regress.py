"""RT-only no-regression gate for the ET-direction change (spec 2026-05-30 §6).

The ET guard never fires without ET periods, so regular-time-only converter
output must be UNCHANGED by this PR. Each golden was captured against silly-kicks
3.30 (pre-4.0.0) by ``capture_goldens.py`` using the identical RT-only input.

Value equality only (``check_dtype=False``): the golden is parquet-roundtripped
(pyarrow backend) while the current run is in-memory, so backend-dtype identity
would be spurious. Each converter's dtype contract is covered by its own
``_finalize_output`` tests (review G).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from regressions.extratime._builders import CASES, run_converter

GOLD = Path(__file__).resolve().parent


def _norm_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Map all null-likes (None / NaN) to a single representation before comparison.

    The golden is parquet-roundtripped (pyarrow turns object-column ``None`` into
    ``NaN``) while the current run is in-memory (``None``). Both are semantically
    null; normalizing to ``NaN`` uniformly makes the value-equality check agnostic
    to null representation (and avoids the pandas None-vs-NaN ``FutureWarning``).
    """
    return df.where(df.notna(), other=np.nan)


@pytest.mark.parametrize("case", CASES)
def test_rt_only_output_value_identical_to_3_30_golden(case):
    golden = pd.read_parquet(GOLD / f"golden_{case}_rt.parquet")
    current = run_converter(case, et=False, flag=None)  # same RT-only input as capture
    # The TF-51-prereq block-detection columns (shot_blocked / cross_blocked) are ADDITIVE and
    # post-date the 3.30 golden; they are not part of this test's existing-value-invariance claim.
    current = current.drop(columns=["shot_blocked", "cross_blocked"], errors="ignore")
    pd.testing.assert_frame_equal(
        _norm_nulls(current.reset_index(drop=True)),
        _norm_nulls(golden.reset_index(drop=True)),
        check_dtype=False,
    )
