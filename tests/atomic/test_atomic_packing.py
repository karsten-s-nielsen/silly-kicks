"""TF-49 atomic packing mirror: numeric columns only, end = x+dx synthesis, and the
SK-xT-2-precedent type-aware result synthesis (dribble intrinsic; pass-class success
iff the next atom is a receival)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.spadl import config as atomicconfig
from silly_kicks.atomic.tracking.features import PackingParams, add_packing, packing_xfns
from tests.tracking.test_defensive_line import _make_frame_rows

_AT = atomicconfig.actiontype_id


def _frame():
    # away defenders at x = 50, 60, 30, 80 for HOME atoms
    return _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
        away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
    )


def _atoms(rows):
    base = {
        "game_id": 1,
        "period_id": 1,
        "team_id": 1,
        "player_id": 50,
        "x": 40.0,
        "y": 34.0,
        "dx": 30.0,
        "dy": 0.0,
        "type_id": _AT["pass"],
    }
    recs = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d["action_id"] = i + 1
        d.setdefault("time_seconds", 1.0 + 0.1 * i)
        recs.append(d)
    return pd.DataFrame(recs)


def test_received_pass_computes_and_numeric_columns_only():
    atoms = _atoms(
        [
            {"time_seconds": 1.0},  # pass 40 -> 70
            {"type_id": _AT["receival"], "x": 70.0, "dx": 0.0, "player_id": 51},
        ]
    )
    out = add_packing(atoms, _frame())
    assert out["packing_made"].iloc[0] == 2  # away 50, 60 in (40, 70]
    assert out["packing_made"].dtype == "Int64"
    assert out["packing_goal_threat"].dtype == "Int64"
    assert out["packing_net"].dtype == np.float64
    # spec s6: receiver/secured omitted; synthesized inputs dropped
    for col in ("packing_receiver_player_id", "packing_secured", "start_x", "start_y", "end_x", "end_y"):
        assert col not in out.columns
    # the receival atom itself is off-domain
    assert pd.isna(out["packing_made"].iloc[1])


def test_unreceived_pass_is_off_domain():
    """Result synthesis discriminator: no receival next -> synthesized fail -> NaN."""
    atoms = _atoms(
        [
            {"time_seconds": 1.0},  # pass 40 -> 70, then an opponent interception atom
            {"type_id": _AT["interception"], "team_id": 2, "x": 70.0, "dx": 0.0, "player_id": 90},
        ]
    )
    out = add_packing(atoms, _frame())
    assert pd.isna(out["packing_made"].iloc[0])


def test_dribble_success_is_intrinsic():
    """Dribbles are never followed by receival -- success must not require one."""
    atoms = _atoms([{"type_id": _AT["dribble"], "time_seconds": 1.0}])
    out = add_packing(atoms, _frame())
    assert out["packing_made"].iloc[0] == 2


def test_degenerate_dribble_nan():
    atoms = _atoms([{"type_id": _AT["dribble"], "dx": 0.0, "dy": 0.0, "time_seconds": 1.0}])
    out = add_packing(atoms, _frame())
    assert pd.isna(out["packing_made"].iloc[0])


def test_require_secured_rejected():
    atoms = _atoms([{}])
    with pytest.raises(ValueError, match="require_secured"):
        add_packing(atoms, _frame(), params=PackingParams(require_secured=True))


def test_output_frame_keeps_caller_type_id_and_gains_no_result_id():
    """Execution-review D3: the adapter's rewritten type_id / synthetic result_id
    must never leak into the returned enrichment frame."""
    atoms = _atoms(
        [
            {"time_seconds": 1.0},
            {"type_id": _AT["receival"], "x": 70.0, "dx": 0.0, "player_id": 51},
        ]
    )
    out = add_packing(atoms, _frame())
    assert "result_id" not in out.columns  # atomic SPADL has none; none may appear
    assert list(out["type_id"]) == list(atoms["type_id"])  # receival atom still visible


def test_collapsed_corner_atom_is_in_domain():
    """Execution-review D2: convert_to_atomic collapses corner_* -> 'corner'; the
    collapsed atom must join the domain when a corner name is requested."""
    atoms = _atoms(
        [
            {"type_id": _AT["corner"], "time_seconds": 1.0},
            {"type_id": _AT["receival"], "x": 70.0, "dx": 0.0, "player_id": 51},
        ]
    )
    out = add_packing(atoms, _frame())
    assert out["packing_made"].iloc[0] == 2


def test_collapsed_freekick_pass_in_domain_but_shot_shape_stays_out():
    """Execution-review D2: a received collapsed 'freekick' counts; one followed by
    a goal atom (the shot_freekick shape) synthesizes fail -> honestly off-domain."""
    atoms = _atoms(
        [
            {"type_id": _AT["freekick"], "time_seconds": 1.0},
            {"type_id": _AT["receival"], "x": 70.0, "dx": 0.0, "player_id": 51, "time_seconds": 1.5},
            {"type_id": _AT["freekick"], "time_seconds": 10.0},
            {"type_id": _AT["goal"], "x": 105.0, "dx": 0.0, "time_seconds": 10.5},
        ]
    )
    out = add_packing(atoms, _frame())
    assert out["packing_made"].iloc[0] == 2
    assert pd.isna(out["packing_made"].iloc[2])


def test_same_team_keeper_pick_up_is_a_completed_reception():
    """Execution-review D5: atomic inserts no receival before keeper collections --
    a completed back-pass to the OWN keeper must still synthesize success, while a
    pass swallowed by the OPPONENT keeper stays fail."""
    back_pass = _atoms(
        [
            {"time_seconds": 1.0, "x": 40.0, "dx": 30.0},
            {"type_id": _AT["keeper_pick_up"], "x": 70.0, "dx": 0.0, "player_id": 51},
        ]
    )
    out = add_packing(back_pass, _frame())
    assert out["packing_made"].iloc[0] == 2

    intercepted = _atoms(
        [
            {"time_seconds": 1.0, "x": 40.0, "dx": 30.0},
            {"type_id": _AT["keeper_pick_up"], "team_id": 2, "x": 70.0, "dx": 0.0, "player_id": 90},
        ]
    )
    out2 = add_packing(intercepted, _frame())
    assert pd.isna(out2["packing_made"].iloc[0])


def test_atomic_xfns_synthesizes_and_emits_nine_columns():
    atoms = _atoms(
        [
            {"time_seconds": 1.0},
            {"type_id": _AT["receival"], "x": 70.0, "dx": 0.0, "player_id": 51},
        ]
    )
    t = packing_xfns()[0]
    cols = t([atoms, atoms, atoms], _frame())
    assert cols.shape == (2, 9)
    assert cols["packing_made_a0"].iloc[0] == 2
    with pytest.raises(ValueError, match="require_secured"):
        packing_xfns(params=PackingParams(require_secured=True))
