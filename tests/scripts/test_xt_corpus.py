"""`scripts/_xt_corpus.py` -- the sharded xT count pass + `fit_from_counts` reduce (spec §4.4).

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import _xt_corpus as xc
import numpy as np
import pandas as pd
import pytest
from _fake_corpus import make_ref

from silly_kicks.xthreat import ExpectedThreat

_L, _W = 16, 12
_SHOT = __import__("silly_kicks.spadl.config", fromlist=["actiontype_id"]).actiontype_id["shot"]
_PASS = __import__("silly_kicks.spadl.config", fromlist=["actiontype_id"]).actiontype_id["pass"]
_SUCCESS = __import__("silly_kicks.spadl.config", fromlist=["result_id"]).result_id["success"]
_FAIL = __import__("silly_kicks.spadl.config", fromlist=["result_id"]).result_id["fail"]


def _row(t, r, sx, sy, ex, ey):
    return {"type_id": t, "result_id": r, "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey}


def _actions(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        _row(_SHOT, _SUCCESS, 20.0, 34.0, 105.0, 34.0),
        _row(_SHOT, _FAIL, 95.0, 34.0, 105.0, 34.0),
        _row(_PASS, _SUCCESS, 10.0, 10.0, 40.0, 20.0),
        _row(_PASS, _SUCCESS, 40.0, 20.0, 70.0, 40.0),
        _row(_PASS, _FAIL, 50.0, 30.0, 65.0, 25.0),
        _row(_PASS, _SUCCESS, 30.0, 60.0, 60.0, 62.0),
    ]
    # a couple of jittered extra moves so different seeds give different counts (additivity is real)
    for _ in range(int(rng.integers(1, 4))):
        rows.append(_row(_PASS, _SUCCESS, float(rng.uniform(10, 90)), float(rng.uniform(5, 60)), 70.0, 40.0))
    return pd.DataFrame(rows)


# ---- Step 1: sparse round-trip -------------------------------------------------------------------


def test_counts_to_frame_from_frames_round_trip():
    counts = ExpectedThreat(l=_L, w=_W).zone_counts(_actions(0))
    frame = xc.counts_to_frame(counts)
    assert list(frame.columns) == xc.COUNT_SHARD_COLUMNS
    assert (frame["n"] > 0).all(), "sparse: only non-zero entries"
    # the four zone aggregates carry to_zone == -1; the transition aggregate carries a real to_zone
    assert (frame.loc[frame["aggregate"] != "transition", "to_zone"] == -1).all()
    assert (frame.loc[frame["aggregate"] == "transition", "to_zone"] >= 0).all()
    back = xc.counts_from_frames([frame], l=_L, w=_W)
    for name in ("shot_counts", "goal_counts", "move_counts", "transition_start_counts", "transition_counts"):
        assert np.array_equal(getattr(back, name), getattr(counts, name)), name


# ---- Step 2: count pass + reduce == pooled fit ---------------------------------------------------


def _pass(tmp_path, seeds, *, allow_failed=False, fail=(), token=None):
    by_key = {("gs", str(s)): _actions(s) for s in seeds}
    refs = [make_ref("gs", str(s)) for s in seeds]
    fail_keys = {("gs", str(s)) for s in fail}

    def _load(ref):
        if ref.key in fail_keys:
            raise RuntimeError(f"boom {ref.key}")
        return by_key[ref.key]

    res = xc.xt_count_pass(
        refs,
        key=lambda r: r.key,
        load_actions=_load,
        shard_root=tmp_path / "sh",
        token_inputs=token or {"schema": "xt-corpus-test-1"},
        l=_L,
        w=_W,
    )
    return res


def test_pass_then_reduce_equals_pooled_fit(tmp_path):
    seeds = [1, 2, 3]
    res = _pass(tmp_path, seeds)
    xt, prov = xc.fit_xt_from_count_pass(res, l=_L, w=_W)
    pooled = pd.concat([_actions(s) for s in seeds], ignore_index=True)
    xt_pooled = ExpectedThreat(l=_L, w=_W).fit(pooled)
    assert np.array_equal(xt.xT, xt_pooled.xT, equal_nan=True)
    assert set(prov.fit_keys) == {"gs__1", "gs__2", "gs__3"} and prov.counts_digest
    assert prov.admission_digest is None and prov.allowed_failed is False and prov.unmeasured_admitted == ()


# ---- Step 3: failure policy ----------------------------------------------------------------------


def test_reduce_refuses_on_failures_by_default(tmp_path):
    res = _pass(tmp_path, [1, 2, 3], fail=[2])
    assert res.failed == 1
    with pytest.raises(RuntimeError, match="failed"):
        xc.fit_xt_from_count_pass(res, l=_L, w=_W)


def test_allow_failed_fits_and_records(tmp_path):
    res = _pass(tmp_path, [1, 2, 3], fail=[2])
    xt, prov = xc.fit_xt_from_count_pass(res, l=_L, w=_W, allow_failed=True)
    assert prov.allowed_failed is True and set(prov.failed) == {"gs__2"}
    # the fit used only the two loaded matches
    pooled = pd.concat([_actions(1), _actions(3)], ignore_index=True)
    assert np.array_equal(xt.xT, ExpectedThreat(l=_L, w=_W).fit(pooled).xT, equal_nan=True)


def test_a_resumed_count_pass_loads_nothing(tmp_path):
    seeds = [1, 2]
    spy = {"loads": 0}
    by_key = {("gs", str(s)): _actions(s) for s in seeds}

    def _load(ref):
        spy["loads"] += 1
        return by_key[ref.key]

    def run():
        return xc.xt_count_pass(
            [make_ref("gs", str(s)) for s in seeds],
            key=lambda r: r.key,
            load_actions=_load,
            shard_root=tmp_path / "sh",
            token_inputs={"schema": "xt-corpus-test-1"},
            l=_L,
            w=_W,
        )

    run()
    assert spy["loads"] == 2
    run()
    assert spy["loads"] == 2, "a resumed count pass re-loaded a finished match"


def test_admission_threads_into_provenance(tmp_path):
    class _Rec:  # structurally an AdmissionRecord (Task 6); typed to match the _AdmissionRecordLike Protocol
        digest: str | None = "abc123"
        unmeasured_admitted: tuple[str, ...] = ("gs__9",)

    res = _pass(tmp_path, [1, 2])
    _xt, prov = xc.fit_xt_from_count_pass(res, l=_L, w=_W, admission=_Rec())
    assert prov.admission_digest == "abc123" and prov.unmeasured_admitted == ("gs__9",)
