"""Structural perf guards for the id-dtype contract (ADR-019).

Not wall-clock (flaky on shared CI). Instead: prove the A1 de-duplication holds -- in the always
cross-dtype path (the lakehouse reality) each hot id column is canonicalized AT MOST ONCE, and the
genuine-string (object x object) path never canonicalizes at all. Benchmarks run but do not gate.
"""

import pandas as pd

from silly_kicks import id_compat as idc
from silly_kicks.tracking.utils import _resolve_action_frame_context
from tests.tracking.conftest_id_dtype import make_actions, make_frames


def _string_frames():
    f = make_frames()
    f["team_id"] = f["team_id"].astype("Int64").astype("string").astype("object")
    f["player_id"] = f["player_id"].astype("Int64").astype("string").astype("object")
    return f


def test_resolve_dedups_canonicalization(monkeypatch):
    # cross-dtype (string frames x numeric actions) + defending_gk so player_id_frame is needed by
    # BOTH the actor and GK masks -- the A1 case the de-dup must collapse to one canonicalization.
    actions = make_actions().assign(defending_gk_player_id=pd.Series([1, 2], dtype="int64"))
    frames = _string_frames()

    import silly_kicks.tracking.utils as U

    seen = []
    real = U.canonical_id_series
    monkeypatch.setattr(
        U,
        "canonical_id_series",
        lambda s: (seen.append(getattr(s, "name", None)), real(s))[1],
    )
    _resolve_action_frame_context(actions, frames)

    # de-dup: no id column canonicalized more than once across the actor/opp/GK masks
    assert len(seen) == len(set(seen)), f"duplicate canonicalization: {seen}"
    assert "player_id_frame" in seen  # the column shared across masks WAS canonicalized (once)


def test_object_object_no_canonicalize_spy(monkeypatch):
    # A1: object x object (genuine-string providers) must take the raw fast path.
    calls = {"n": 0}
    real = idc.canonical_id_series
    monkeypatch.setattr(
        idc,
        "canonical_id_series",
        lambda s: (calls.__setitem__("n", calls["n"] + 1), real(s))[1],
    )
    a = pd.Series(["DFL-A", "DFL-B"] * 50000, dtype="object")
    b = pd.Series(["DFL-A", "DFL-Z"] * 50000, dtype="object")
    idc.ids_equal(a, b)
    idc.ids_differ(a, b)
    assert calls["n"] == 0


def test_resolve_cross_dtype_benchmark(benchmark):
    actions = make_actions()
    frames = pd.concat([_string_frames()] * 2000, ignore_index=True)
    benchmark(lambda: _resolve_action_frame_context(actions.copy(), frames.copy()))


def test_resolve_object_object_benchmark(benchmark):
    # genuine-string provider path (sportec/kloppy): both sides object.
    actions = make_actions()
    actions["team_id"] = actions["team_id"].astype(str)
    actions["player_id"] = actions["player_id"].astype(str)
    frames = pd.concat([_string_frames()] * 2000, ignore_index=True)
    benchmark(lambda: _resolve_action_frame_context(actions.copy(), frames.copy()))
