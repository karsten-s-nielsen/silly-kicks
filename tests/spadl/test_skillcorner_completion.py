"""SkillCorner native-completion result_id fix (D-S8, reviews N1/F1/G1).

Single-construct pass completion: pass_outcome PRIMARY, received==True success-ONLY
(received==False NEVER -> fail, N1), residual -> flagged stopgap. result_source tiers
{native, inferred, stopgap} drive the training filter (G1: train on native only)."""

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.skillcorner import _native_completion_result

S, F = spadlconfig.result_id["success"], spadlconfig.result_id["fail"]


def test_pass_outcome_primary_and_received_success_only():
    # rows: 0 native success, 1 native unsuccessful, 2 offside->fail, 3 no pass_outcome+received=True,
    # 4 no pass_outcome + received=False but same_team_next=True (N1: NOT forced fail -> residual),
    # 5 neither native field, same_team_next=False
    pass_outcome = pd.Series(["successful", "unsuccessful", "offside", np.nan, np.nan, np.nan])
    received = pd.Series([np.nan, np.nan, np.nan, True, False, np.nan])
    same_team_next = pd.Series([True, True, False, False, True, False])
    is_passlike = pd.Series([True, True, True, True, True, True])
    rid, src = _native_completion_result(pass_outcome, received, same_team_next, is_passlike)
    assert rid[0] == S and src[0] == "native"  # pass_outcome successful
    assert rid[1] == F and src[1] == "native"  # unsuccessful
    assert rid[2] == F and src[2] == "native"  # offside -> fail
    assert rid[3] == S and src[3] == "inferred"  # received True -> success (clean signal, NOT native)
    # N1 regression: row 4 has received=False but is NOT routed to fail by received;
    # it falls to the residual (same_team_next=True here) -> success, tagged stopgap.
    assert rid[4] == S and src[4] == "stopgap"
    assert rid[5] == F and src[5] == "stopgap"


def test_non_passlike_rows_untouched():
    # A non-passlike row (e.g. a clearance handled elsewhere) keeps the fail default + stopgap tag here;
    # the converter's np.select gives it an explicit result, so _native_completion_result only owns passlike.
    pass_outcome = pd.Series(["successful"])
    received = pd.Series([np.nan])
    same_team_next = pd.Series([True])
    is_passlike = pd.Series([False])
    rid, src = _native_completion_result(pass_outcome, received, same_team_next, is_passlike)
    assert rid[0] == F  # default (the converter overrides non-passlike rows with their own result)
    assert src[0] == "stopgap"


# --- Integration: through convert_to_actions on the committed fixture ---

import json  # noqa: E402
from pathlib import Path  # noqa: E402

_FIXTURE_DIR = Path(__file__).parent.parent / "datasets" / "skillcorner"


def _load_basic():
    events = pd.read_csv(_FIXTURE_DIR / "basic_possessions.csv")
    with open(_FIXTURE_DIR / "match_metadata.json") as f:
        meta = json.load(f)
    return events, meta


def test_converter_emits_valid_result_source():
    # The committed fixture has no pass_outcome/received cols -> passlike rows take the stopgap path
    # (backward-compatible with the old same_team_next behavior); clearance/foul/shot are tagged native.
    from silly_kicks.spadl.skillcorner import convert_to_actions

    events, meta = _load_basic()
    actions, _ = convert_to_actions(events, meta)
    assert "result_source" in actions.columns
    assert set(actions["result_source"].dropna().unique()) <= {"native", "inferred", "stopgap"}


def test_native_pass_outcome_overrides_stopgap_and_preserves_explicit_results():
    # D-S8: injecting a native pass_outcome on the pass rows promotes them stopgap -> native;
    # Chesterton scope (Task 3): clearance/foul/shot result_id is unaffected by the pass_outcome column.
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.spadl.skillcorner import convert_to_actions

    events, meta = _load_basic()
    base, _ = convert_to_actions(events, meta)
    base_native = int((base["result_source"] == "native").sum())  # clearance/foul/shot only

    ev2 = events.copy()
    ev2["pass_outcome"] = pd.Series(
        ["successful" if t == "pass" else None for t in ev2["end_type"]],
        index=ev2.index,
        dtype=object,
    )
    inj, _ = convert_to_actions(ev2, meta)

    # pass rows promoted stopgap -> native
    assert int((inj["result_source"] == "native").sum()) > base_native
    native_rows = inj[inj["result_source"] == "native"]
    assert (native_rows["result_id"] == spadlconfig.result_id["success"]).any()  # injected 'successful'

    # Chesterton scope guard: deterministic clearance/foul/shot result_id unchanged by the injection.
    # In the baseline (no pass_outcome), passlike rows are "stopgap"; the "native"-tagged rows are the
    # explicit clearance/foul/shot branches -> their result_id must survive the injection unchanged.
    det = inj["result_source"] == "native"
    base_det = base["result_source"] == "native"
    assert set(base.loc[base_det, "result_id"]).issubset(set(inj.loc[det, "result_id"]))
