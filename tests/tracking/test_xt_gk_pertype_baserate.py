"""compute_xt_gk per-type base-rate serve switch (xt_gk_completion_source).

compute_xt_gk returns ONLY the xt_gk value + provenance columns (no type_id), indexed like the
input actions -- so type filters use the input actions' type_id (positional / index-aligned).
"""

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._xt_gk import compute_xt_gk
from tests.tracking.test_gk_completion_pertype_gate import _fitted_model_with_gate as _gate_model
from tests.tracking.test_xt_gk import _frames_for, _gk_actions, _gk_realistic_xt

_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_THROW_IN = spadlconfig.actiontype_id["throw_in"]


def test_goalkick_gated_to_base_rate_is_tagged_and_differs_from_model():
    actions, xt = _gk_actions(), _gk_realistic_xt()
    frames = _frames_for(actions)
    gated = _gate_model({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    model = _gate_model({"goalkick": "model", "throw_in": "model", "other": "model"})
    out_b = compute_xt_gk(actions, frames, xt=xt, completion=gated)
    out_m = compute_xt_gk(actions, frames, xt=xt, completion=model)
    is_gk = (actions["type_id"] == _GOALKICK).to_numpy()
    assert (out_b[is_gk]["xt_gk_completion_source"] == "base_rate").all()  # tagged
    # the back-pass (other) stays "model" in both
    other = out_b[~is_gk & out_b["xt_gk_completion_source"].notna().to_numpy()]
    assert (other["xt_gk_completion_source"] == "model").all()
    # base-rate p flows into RAV -> the goal-kick xt_gk differs from the model-scored value
    assert not np.allclose(out_b[is_gk]["xt_gk"].to_numpy(), out_m[is_gk]["xt_gk"].to_numpy(), equal_nan=True)


def test_switch_is_noop_when_no_type_gated():
    # Regression lock (review L2): when no type is gated "base_rate", the switch never overrides pc,
    # so it is a provable no-op. (Compares an explicit all-"model" gate vs the fail-open empty gate --
    # both new-code paths; the pre-switch code is gone and can't be run, so this asserts the no-op
    # property, not literal new-vs-old byte-identity.)
    actions, xt = _gk_actions(), _gk_realistic_xt()
    frames = _frames_for(actions)
    gated = _gate_model({"goalkick": "model", "throw_in": "model", "other": "model"})
    ungated = _gate_model({})  # fail-open -> all model
    a = compute_xt_gk(actions, frames, xt=xt, completion=gated)
    b = compute_xt_gk(actions, frames, xt=xt, completion=ungated)
    np.testing.assert_allclose(a["xt_gk"].to_numpy(), b["xt_gk"].to_numpy(), atol=1e-12, equal_nan=True)
    assert (a["xt_gk_completion_source"].dropna() == "model").all()


def test_throw_in_gated_to_base_rate_is_tagged():
    # review M1: a throw-in (degenerate positive class -> base_rate gate) must also be tagged base_rate
    # at the switch, not just goal-kicks. Build a GK throw-in at an EXISTING frame time so it links
    # (review-3 L-A): _frames_for emits frames only at 5.0/50.0, so the throw-in uses t=50.0 (the GK,
    # player 10, sits at (5,34) in both frames) and the unmodified _frames_for(_gk_actions()).
    base = _gk_actions().iloc[[0]].copy()  # the goalkick row's shape (GK actor, finite coords)
    base["action_id"] = [2]
    base["type_id"] = [_THROW_IN]
    base["time_seconds"] = [50.0]  # an existing frame time -> the throw-in links + resolves geometry
    actions = pd.concat([_gk_actions(), base], ignore_index=True)
    frames = _frames_for(_gk_actions())  # frames at 5.0 + 50.0 (hard-coded; independent of action times)
    gated = _gate_model({"goalkick": "model", "throw_in": "base_rate", "other": "model"})
    out = compute_xt_gk(actions, frames, xt=_gk_realistic_xt(), completion=gated)
    is_ti = (actions["type_id"] == _THROW_IN).to_numpy()
    assert is_ti.sum() == 1 and (out[is_ti]["xt_gk_completion_source"] == "base_rate").all()


def test_report_completion_source_counts():
    from silly_kicks.tracking._xt_gk import XtGkReport

    actions, xt = _gk_actions(), _gk_realistic_xt()
    frames = _frames_for(actions)
    model = _gate_model({"goalkick": "base_rate", "throw_in": "base_rate", "other": "model"})
    out = compute_xt_gk(actions, frames, xt=xt, completion=model)
    rep = XtGkReport.from_frame(out)
    vc = out["xt_gk_completion_source"].value_counts(dropna=True)
    assert rep.completion_source_counts == {str(k): int(v) for k, v in vc.items()}


def test_atomic_path_inherits_base_rate_switch(monkeypatch):
    # add_xt_gk does NOT thread completion= (features.py:5177), so inject the gate by monkeypatching
    # from_variant. Build atomic input via the column-rename scaffold (test_xt_gk.py:608-612) which
    # PRESERVES type_id -- NOT convert_to_atomic (which remaps the type enumeration -> breaks ==_GOALKICK).
    import silly_kicks.tracking._gk_completion as gkc
    from silly_kicks.atomic.tracking.features import add_xt_gk as atomic_add_xt_gk
    from silly_kicks.tracking.features import add_xt_gk as std_add_xt_gk

    gated = _gate_model({"goalkick": "base_rate", "throw_in": "model", "other": "model"})
    monkeypatch.setattr(gkc.GkCompletionModel, "from_variant", classmethod(lambda cls, variant="default": gated))
    std = _gk_actions()
    frames = _frames_for(std)
    atom = std.rename(columns={"start_x": "x", "start_y": "y"}).copy()
    atom["dx"] = std["end_x"].to_numpy() - std["start_x"].to_numpy()
    atom["dy"] = std["end_y"].to_numpy() - std["start_y"].to_numpy()
    atom = atom.drop(columns=["end_x", "end_y"])
    atom_out = atomic_add_xt_gk(atom, frames, _gk_realistic_xt(), home_team_id=1)
    std_out = std_add_xt_gk(std, frames, _gk_realistic_xt(), home_team_id=1)
    is_gk = (std["type_id"] == _GOALKICK).to_numpy()
    assert is_gk.sum() >= 1 and (atom_out[is_gk]["xt_gk_completion_source"] == "base_rate").all()  # inherited
    # parity: atomic mirror tags identically to the standard path (mirrors test_xt_gk.py:629-632)
    assert (
        atom_out["xt_gk_completion_source"].to_numpy().tolist()
        == std_out["xt_gk_completion_source"].to_numpy().tolist()
    )
