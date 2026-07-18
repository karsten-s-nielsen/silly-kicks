"""ADR-019 primary gate: feature outputs must not depend on id dtype.

The production failure is ASYMMETRIC (numeric actions x string frames). Casting BOTH sides to
string would be homogeneous and pass on broken code (object==object works) -- so we vary the two
sides INDEPENDENTLY and assert every permutation equals the all-numeric baseline. home_team_id
dtype is a SEPARATE axis.
"""

import pandas as pd
import pytest

from tests.tracking.conftest_id_dtype import (
    AGGREGATORS,
    NON_LINKED_AGGREGATORS,
    REGISTERED_AGGREGATORS,
    make_actions,
    make_frames,
)

# ADR-041 opt-out: auto-enumerating gate -- it sweeps EVERY registered aggregator on defaults, so the OBSO
# family's synthetic-EPV notice is expected here and unrelated to what this gate asserts.
pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")

# Entity-id columns we CAST when building the asymmetric variants (input side).
STRINGIFY_COLS = ["team_id", "player_id", "defending_gk_player_id"]

# N-e: aggregators return rows deterministically ordered by action_id (the SPADL contract),
# so baseline and variant align positionally after reset_index -- a same-rows-different-order
# variant would (correctly) be treated as a regression. Assumption stated, not silently relied on.


def _is_id_col(name: str) -> bool:
    """B1: any id-valued OUTPUT column must be excluded from the value comparison -- a numeric
    baseline (99) vs a string variant ("99") legitimately differ, and assert_frame_equal(
    check_dtype=False) still compares VALUES. Excluding only team_id/player_id is too narrow."""
    return "team_id" in name or "player_id" in name or name.endswith("_id")


def _stringify(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64").astype("string").astype("object")
    return df


# (actions_to_string, frames_to_string, home_team_id_to_string)
PERMUTATIONS = [
    (False, True, False),  # numeric actions x STRING frames  <- the lakehouse bug
    (True, False, False),  # STRING actions x numeric frames  <- the reverse
    (False, False, True),  # string home_team_id only         <- scalar-arg axis
    (True, True, True),  # all string                       <- homogeneous sanity
]


@pytest.mark.parametrize("agg", AGGREGATORS, ids=lambda a: a.__name__)
@pytest.mark.parametrize("act_str,frm_str,ht_str", PERMUTATIONS)
def test_aggregator_id_dtype_invariant(agg, act_str, frm_str, ht_str):
    base_actions, base_frames, home = make_actions(), make_frames(), 5
    baseline = agg(base_actions.copy(), base_frames.copy(), home)

    a = _stringify(base_actions, STRINGIFY_COLS) if act_str else base_actions.copy()
    f = _stringify(base_frames, STRINGIFY_COLS) if frm_str else base_frames.copy()
    h = "5" if ht_str else home
    variant = agg(a, f, h)

    # compare FEATURE columns only -- every id-valued column legitimately differs
    # numeric-vs-string and must be excluded (B1, generic id-name rule).
    feat_cols = [c for c in baseline.columns if not _is_id_col(c)]
    pd.testing.assert_frame_equal(
        baseline[feat_cols].reset_index(drop=True),
        variant[feat_cols].reset_index(drop=True),
        check_dtype=False,
        check_like=True,
    )


def test_enumerated_surface_equals_registered():  # B3 meta-assertion
    enumerated = {a.__name__ for a in AGGREGATORS}
    covered = enumerated | set(NON_LINKED_AGGREGATORS)
    assert covered == REGISTERED_AGGREGATORS, (
        "id-dtype gate must cover every registered public aggregator (in AGGREGATORS or, "
        "with a justification, NON_LINKED_AGGREGATORS); "
        f"uncovered: {REGISTERED_AGGREGATORS - covered}"
    )
