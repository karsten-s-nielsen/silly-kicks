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
    # The TF-46 Arm 1 base signal columns (result_source / restart_evidence / type_source) are
    # ADDITIVE and post-date the 3.30 golden; each declares converter provenance and changes no
    # existing value, so they are carved out exactly like the block-detection columns above.
    current = current.drop(columns=["result_source", "restart_evidence", "type_source"], errors="ignore")
    # TF-46 Arm 1 Commit 3 ADDED action_provenance + ball_claim_type to SPORTEC_SPADL_COLUMNS (decision B2);
    # both post-date the 3.30 golden and are additive (they declare row provenance / a DFL claim qualifier
    # and change no existing value), so they are carved out exactly like the base signal columns above.
    current = current.drop(columns=["action_provenance", "ball_claim_type"], errors="ignore")
    # TF-46 Arm 1 Part 2 ADDED penalty_result_method + shot_woodwork_type to SPORTEC_SPADL_COLUMNS (§25 DFL
    # Penalty mapping / BallDeflection routing) and promoted repair_kind to BASE SPADL_COLUMNS (decision D1).
    # All post-date the 3.30 golden and are additive (penalty result-method / woodwork token / reconciliation
    # provenance, changing no existing value), so they are carved out exactly like the base signal columns above.
    current = current.drop(columns=["penalty_result_method", "shot_woodwork_type", "repair_kind"], errors="ignore")
    # TF-46 Arm 1 Part 2 ADDED six GI facts to SKILLCORNER_SPADL_COLUMNS (spec section 17.2, CL-1 / CL-13 /
    # B-11): the bracket frame ids and the possession facts carried onto GI-backed rows. All post-date the
    # 3.30 golden and are additive (GI possession provenance, changing no existing value), so they are carved
    # out exactly like the base signal columns above.
    current = current.drop(
        columns=[
            "gi_marker_possession_frame_end",
            "gi_next_possession_frame_start",
            "gi_attacking_side",
            "gi_possession_frame_start",
            "gi_possession_frame_end",
            "gi_event_id",
        ],
        errors="ignore",
    )
    # ADR-103 (4.125.0) DROPPED the all-null `confidence` frame column; it is present in the 3.30
    # golden and gone from current output. Removing a column that was uniformly null is not a value
    # change, so it is carved out of the invariance claim (mirrors the additive carve-out above).
    golden = golden.drop(columns=["confidence"], errors="ignore")
    pd.testing.assert_frame_equal(
        _norm_nulls(current.reset_index(drop=True)),
        _norm_nulls(golden.reset_index(drop=True)),
        check_dtype=False,
        # ADR-103 set the low-card frame columns (ball_state / source_provider / is_goalkeeper_source)
        # to `category`; the golden is parquet-roundtripped as `object`. Same rationale as check_dtype:
        # backend/dtype identity is spurious here, this test asserts VALUE equality only.
        check_categorical=False,
    )
