"""TF-49 golden identity: packing_made == structural_lbs on completed pass/cross rows.

The ~15-line defender-extraction/mirror block is deliberately duplicated from
_structural_pass.py (frozen-kernel isolation, ADR-039); this gate pins the two
copies to byte-equivalent counts. The completion gate is the ONLY delta on the
(pass, cross) slice -- the discriminator row proves it (mutating the gate out
turns this red). Non-vacuity meta-assertions per the deep-zone-gate precedent
(review major 6): an all-NaN or all-zero comparison must FAIL, not pass silently.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import PackingParams, add_packing, add_structural_pass
from tests.tracking.test_defensive_line import _make_frame_rows

_R = spadlconfig.result_id
_T = spadlconfig.actiontype_id


def _frame():
    # away defenders at x = 50, 60, 30, 80 for HOME actions
    return _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
        away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        time_seconds=10.0,
    )


def _actions():
    base = {
        "game_id": 1,
        "period_id": 1,
        "time_seconds": 10.0,
        "team_id": 1,
        "player_id": 50,
        "start_y": 34.0,
        "end_y": 34.0,
        "result_id": _R["success"],
    }
    rows = [
        {"type_id": _T["pass"], "start_x": 40.0, "end_x": 70.0},  # made 2 (50, 60)
        {"type_id": _T["cross"], "start_x": 55.0, "end_x": 65.0},  # made 1 (60)
        {"type_id": _T["pass"], "start_x": 40.0, "end_x": 45.0},  # made 0 (honest zero)
        {"type_id": _T["pass"], "start_x": 40.0, "end_x": 70.0, "result_id": _R["fail"]},  # discriminator
    ]
    recs = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d["action_id"] = i
        recs.append(d)
    return pd.DataFrame(recs)


def test_packing_made_equals_structural_lbs_on_completed_pass_cross():
    a, f = _actions(), _frame()
    packed = add_packing(a, f, home_team_id=1, params=PackingParams(action_types=("pass", "cross")))
    struct = add_structural_pass(a, f, home_team_id=1)

    completed = (a["result_id"] == _R["success"]).to_numpy()
    both = packed["packing_made"].notna().to_numpy() & struct["structural_lbs"].notna().to_numpy() & completed

    # Non-vacuity meta-assertions (review major 6, deep-zone-gate precedent).
    assert int(both.sum()) >= 3, "golden gate vacuous: need >= 3 comparable completed rows"
    pm = packed.loc[both, "packing_made"].reset_index(drop=True)
    sl = struct.loc[both, "structural_lbs"].reset_index(drop=True)
    assert (pm > 0).any(), "golden gate vacuous: all-zero comparison proves nothing"

    pd.testing.assert_series_equal(pm, sl, check_names=False)


def test_completion_gate_is_the_only_delta():
    """The failed pass has numeric structural_lbs (no result gate in TF-45) but
    <NA> packing_made -- mutating packing's completion gate out turns this red."""
    a, f = _actions(), _frame()
    packed = add_packing(a, f, home_team_id=1, params=PackingParams(action_types=("pass", "cross")))
    struct = add_structural_pass(a, f, home_team_id=1)

    failed = (a["result_id"] == _R["fail"]).to_numpy()
    disc = failed & struct["structural_lbs"].notna().to_numpy() & packed["packing_made"].isna().to_numpy()
    assert disc.any(), "no failed row separates the two kernels -- discriminator lost"
