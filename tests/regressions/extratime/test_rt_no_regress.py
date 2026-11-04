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
    # ADR-125 (4.137.0) ADDED the `z_source` ball-z provenance column to the tracking-frame schema
    # (tracking cases only); it post-dates the 3.30 golden and is additive (declares provenance, changes
    # no existing value), so it is carved out exactly like the block-detection columns above.
    current = current.drop(columns=["z_source"], errors="ignore")
    # ADR-103 (4.125.0) DROPPED the all-null `confidence` frame column; it is present in the 3.30
    # golden and gone from current output. Removing a column that was uniformly null is not a value
    # change, so it is carved out of the invariance claim (mirrors the additive carve-out above).
    golden = golden.drop(columns=["confidence"], errors="ignore")
    # TF-46 Arm-1 Part 1/2 ADDED provenance columns absent from the 3.30 golden, all additive (new
    # provenance, no existing value changes): Part 1 (§11.2) result_source / restart_evidence /
    # type_source and Part 2 repair_kind (D1) on every event converter; plus, for sportec_events only,
    # action_provenance / ball_claim_type / penalty_result_method / shot_woodwork_type (§25). The expected
    # set is PER CASE (gs/metrica = 4, sportec = 8, tracking = none), not one blanket allow-list: a per-case
    # set also catches a slice column that VANISHES from one case or WRONGLY appears in another (B11). The
    # two asserts below are the anti-drift guard -- every expected column must be present-and-new, and no
    # other new column may appear -- so a real future drop/rename/leak surfaces instead of being swallowed.
    _EVENT_SLICE = {"result_source", "restart_evidence", "type_source", "repair_kind"}
    _SLICE_ADDITIVE_BY_CASE = {
        "sportec_tracking": set(),
        "gs_tracking": set(),
        "gs_events": set(_EVENT_SLICE),
        "metrica_events": set(_EVENT_SLICE),
        "sportec_events": _EVENT_SLICE
        | {"action_provenance", "ball_claim_type", "penalty_result_method", "shot_woodwork_type"},
    }
    expected = _SLICE_ADDITIVE_BY_CASE[case]
    extra = set(current.columns) - set(golden.columns)
    unaccounted = sorted(extra - expected)
    assert not unaccounted, f"unaccounted new column(s) vs the 3.30 golden [{case}]: {unaccounted}"
    missing = sorted(expected - extra)
    assert not missing, f"expected slice-additive column(s) absent from current [{case}]: {missing}"
    current = current.drop(columns=list(expected), errors="ignore")
    pd.testing.assert_frame_equal(
        _norm_nulls(current.reset_index(drop=True)),
        _norm_nulls(golden.reset_index(drop=True)),
        check_dtype=False,
        # ADR-103 set the low-card frame columns (ball_state / source_provider / is_goalkeeper_source)
        # to `category`; the golden is parquet-roundtripped as `object`. Same rationale as check_dtype:
        # backend/dtype identity is spurious here, this test asserts VALUE equality only.
        check_categorical=False,
    )
