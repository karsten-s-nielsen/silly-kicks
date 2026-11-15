"""kloppy: ``to_spadl_ltr`` moved before ``_add_dribbles`` (ADR-133, spec §6.1 run position).

The delivery rule needs canonical SPADL LTR rows before synthetic dribbles exist. In the kloppy converter
``to_spadl_ltr`` ran after ``_add_dribbles``. The move must not change any output except the new delivery
typing: ``_add_dribbles`` pairs only same-team, same-period rows, and ``to_spadl_ltr`` mirrors a team's rows
in a period as one block. The golden frame was captured from the pre-move converter (``0cc07e6c`` plus
the all-``<NA>`` ``cross_native`` placeholder) with ``type_id`` and ``cross_native`` dropped; ``time_source``
(spec r8 §6.6 C) is dropped too, so the golden keeps its pre-move provenance.

The SkillCorner case retired in commit 3 (spec r8 §6.6 A, review A CD-PLAN-08). It proved the same move for
SkillCorner against its own pre-move golden, and commit 1 (``4490016c``) holds that proof. Section 6.6 A then
moved every SkillCorner native row to the kick on purpose, so a golden regenerated from the r8 converter would
only snapshot the code under test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.spadl._converter_cases import CONVERTER_CASES

GOLDEN = Path(__file__).parent / "fixtures" / "reorder_golden"
_EXCLUDED = ["type_id", "cross_native", "time_source"]


def _output(name: str) -> pd.DataFrame:
    case = CONVERTER_CASES[name]
    out = case.run(case.build_input())
    return out.drop(columns=[c for c in _EXCLUDED if c in out.columns]).reset_index(drop=True)


@pytest.mark.parametrize("name", ["kloppy"])
def test_reorder_is_byte_identical(name):
    pytest.importorskip("kloppy")
    golden = pd.read_parquet(GOLDEN / f"{name}.parquet")
    # Parquet round-trips string columns as StringDtype on pandas 3, and a missing object value as nan where the
    # converter emits <NA>. The dtypes are pinned by _finalize_output, so compare the values exactly on an object
    # view with one missing-value spelling.
    pd.testing.assert_frame_equal(_norm(_output(name)), _norm(golden), check_exact=True)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(object).map(lambda v: None if pd.isna(v) else v)


if __name__ == "__main__":  # regenerate the golden frame (pre-move converter only)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    _output("kloppy").to_parquet(GOLDEN / "kloppy.parquet", index=False)
    print("wrote", GOLDEN / "kloppy.parquet")
