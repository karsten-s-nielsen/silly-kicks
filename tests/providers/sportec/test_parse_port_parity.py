"""Golden parity for the DFL parse+shape port (PR-S95 / ADR-031 T3).

Asserts the upstreamed ``silly_kicks.providers.sportec`` port reproduces the GENUINE
luxury-lakehouse @ ``0efac60`` parser/shaper output, captured by running the REAL lakehouse
functions on a reduced real-WC2022 IDSSE slice (Phase 0; see
``tests/datasets/sportec/idsse_slice/SOURCE_SHA``). This is the guard against any
transcription / adaptation error in the verbatim lift -- a genuine
"port reproduces production" check, not copy-vs-copy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.providers.sportec import (
    parse_dfl_events,
    parse_dfl_match_info,
    parse_dfl_tracking,
    shape_events_to_native,
    shape_tracking_to_native,
)

_FIX = Path(__file__).resolve().parents[2] / "datasets" / "sportec" / "idsse_slice"
_MATCH_ID = "J03WMX"  # bare DFL MatchId (DFL-MAT-J03WMX -> J03WMX)


def _assert_parity(got: pd.DataFrame, golden_name: str, *, sort_keys: list[str]) -> None:
    """Compare a port DataFrame to a committed golden parquet, order-agnostic.

    The bronze column SET + per-cell values are the seam contract (ADR-031 N1); incidental
    column / row ORDER is not (``finalize_bronze_df`` appends missing cols, and the cross-repo
    contract is by name). So align columns by name and sort rows by a stable key before
    comparing. Float columns are compared with a tiny tolerance (same parser code, same input
    -> bit-identical bar float-repr noise); dtypes are part of the contract.
    """
    golden = pd.read_parquet(_FIX / f"{golden_name}.parquet")
    assert set(got.columns) == set(golden.columns), (
        f"{golden_name}: column-set drift\n  port-only: {sorted(set(got.columns) - set(golden.columns))}"
        f"\n  golden-only: {sorted(set(golden.columns) - set(got.columns))}"
    )
    cols = sorted(golden.columns)
    g = got[cols].sort_values(sort_keys, na_position="last").reset_index(drop=True)
    e = golden[cols].sort_values(sort_keys, na_position="last").reset_index(drop=True)
    # Canonicalise string-like columns to object-with-None on BOTH sides. The seam contract is the
    # column SET + per-cell VALUES, NOT the pandas-version-specific string dtype: pandas 2 (py3.10)
    # infers string columns as ``object`` while pandas 3 (py3.11+) infers ``str``/``StringDtype``,
    # and the parquet round-trip can surface the parser's Python ``None`` as ``nan``. Normalising
    # to ``object`` + ``None`` makes the value comparison cross-version stable (and silences the
    # pandas nan-vs-None FutureWarning). ``check_dtype=False`` for the same reason -- a wrong-typed
    # value (e.g. int ``366`` vs str ``"366"``) is still caught by the value comparison; the nullable
    # numeric dtypes are produced by the verbatim-lifted ``finalize_bronze_df`` on both sides.
    for df in (g, e):
        for c in df.columns:
            dt = str(df[c].dtype)
            if df[c].dtype == object or dt == "str" or dt.startswith("string"):
                df[c] = [None if pd.isna(v) else v for v in df[c]]
    pd.testing.assert_frame_equal(g, e, check_exact=False, atol=1e-9, rtol=0.0, check_dtype=False)


@pytest.fixture(scope="module")
def match_info():
    return parse_dfl_match_info(str(_FIX / "info.xml"))


def test_match_info_basic(match_info):
    assert match_info.home_team_id == "DFL-CLU-000008"
    assert match_info.away_team_id == "DFL-CLU-00000G"
    assert match_info.competition_id == "DFL-COM-000001"
    assert match_info.player_team_map  # non-empty roster
    assert match_info.gk_player_ids  # at least one GK


def test_tracking_bronze_matches_lakehouse_golden(match_info):
    bronze = parse_dfl_tracking(str(_FIX / "positions.xml"), match_info=match_info, match_id=_MATCH_ID)
    _assert_parity(bronze, "idsse_parse_bronze_golden", sort_keys=["frame", "player_id"])


def test_tracking_native_matches_lakehouse_golden(match_info):
    bronze = parse_dfl_tracking(str(_FIX / "positions.xml"), match_info=match_info, match_id=_MATCH_ID)
    native = shape_tracking_to_native(bronze)
    _assert_parity(native, "idsse_shape_native_golden", sort_keys=["frame_id", "is_ball", "player_id"])


def test_events_bronze_matches_lakehouse_golden(match_info):
    bronze = parse_dfl_events(str(_FIX / "events.xml"), match_info=match_info, match_id=_MATCH_ID)
    _assert_parity(bronze, "idsse_events_bronze_golden", sort_keys=["event_id"])


def test_events_native_matches_lakehouse_golden(match_info):
    bronze = parse_dfl_events(str(_FIX / "events.xml"), match_info=match_info, match_id=_MATCH_ID)
    native = shape_events_to_native(bronze)
    _assert_parity(native, "idsse_events_native_golden", sort_keys=["event_id"])
