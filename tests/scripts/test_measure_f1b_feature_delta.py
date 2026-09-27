"""Unit + end-to-end tests for scripts/measure_f1b_feature_delta.py (ADR-106 T10).

The driver's SCIENCE (real per-feature float32 deltas) is validated on the owner corpus; here we
pin the pure plumbing -- the two-arm cast, the type_name restore, the per-feature delta math, the
status paths, the corpus rollup, and that main() writes a provenance-stamped metrics.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.measure_f1b_feature_delta import (
    _ATOL,
    _aggregate,
    _arm,
    _delta_rows,
    _with_type_name,
)


def test_arm_casts_coords_and_preserves_nan():
    df = pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0], "other": [9.0, 9.0]})
    f32 = _arm(df, "float32")
    assert f32["x"].dtype == np.float32 and f32["y"].dtype == np.float32
    assert f32["other"].dtype == np.float64  # non-coord column untouched
    assert bool(np.isnan(f32["x"].to_numpy()[1]))  # NaN preserved through the cast
    assert _arm(df, "float64")["x"].dtype == np.float64


def test_with_type_name_maps_type_id():
    import silly_kicks.spadl.config as spadlconfig

    pass_id = spadlconfig.actiontype_id["pass"]
    out = _with_type_name(pd.DataFrame({"type_id": [pass_id]}))
    assert out is not None
    assert out["type_name"].iloc[0] == "pass"


def test_with_type_name_noop_when_present_or_absent():
    have = pd.DataFrame({"type_id": [0], "type_name": ["keep"]})
    kept = _with_type_name(have)
    assert kept is not None and kept["type_name"].iloc[0] == "keep"  # not overwritten
    absent = _with_type_name(pd.DataFrame({"foo": [1]}))
    assert absent is not None and "type_name" not in absent.columns  # no type_id -> no col
    assert _with_type_name(None) is None


def _stub_adapter(seq):
    """An adapter returning `seq[i]` on its i-th call (arm f64 then f32)."""
    calls = {"n": 0}

    def adapter(_frames, _actions, _home):
        i = calls["n"]
        calls["n"] += 1
        return seq[i]

    return adapter


def test_delta_rows_computes_per_feature_delta():
    f64 = pd.DataFrame({"f": [1.0, 2.0, 3.0], "g": [0.0, 0.0, 0.0]})
    f32 = pd.DataFrame({"f": [1.0, 2.0 + 1e-3, 3.0], "g": [0.0, 0.0, 0.0]})
    rows = _delta_rows("m", "mdl", _stub_adapter([f64, f32]), None, None, None, None)
    by = {r["feature"]: r for r in rows}
    assert by["f"]["status"] == "ok" and by["f"]["n_rows"] == 3
    assert by["f"]["max_abs"] == pytest.approx(1e-3)
    assert by["f"]["n_gt_atol"] == 1  # only row 1 exceeds atol
    assert by["g"]["max_abs"] == 0.0 and by["g"]["n_gt_atol"] == 0


def test_delta_rows_status_paths():
    empty = _delta_rows("m", "mdl", _stub_adapter([None, None]), None, None, None, None)
    assert len(empty) == 1 and empty[0]["status"] == "empty"

    mismatch = _delta_rows(
        "m", "mdl", _stub_adapter([pd.DataFrame({"f": [1.0]}), pd.DataFrame({"f": [1.0, 2.0]})]), None, None, None, None
    )
    assert mismatch[0]["status"] == "row_mismatch"

    def boom(_f, _a, _h):
        raise ValueError("nope")

    err = _delta_rows("m", "mdl", boom, None, None, None, None)
    assert err[0]["status"].startswith("error:ValueError")


def test_delta_rows_ignores_nonfinite_pairs():
    f64 = pd.DataFrame({"f": [1.0, np.nan, 3.0]})
    f32 = pd.DataFrame({"f": [1.0 + 2e-3, np.nan, 3.0]})
    (row,) = _delta_rows("m", "mdl", _stub_adapter([f64, f32]), None, None, None, None)
    assert row["n_rows"] == 2  # the NaN pair is dropped
    assert row["max_abs"] == pytest.approx(2e-3)


def test_aggregate_rolls_up_counts_and_classifies():
    combined = pd.DataFrame(
        [
            # model mdl, feature moved: two matches summed
            dict(
                match_key="a",
                model="mdl",
                feature="moved",
                status="ok",
                n_rows=10,
                sum_abs=1.0,
                max_abs=5e-3,
                n_gt_atol=4,
            ),
            dict(
                match_key="b",
                model="mdl",
                feature="moved",
                status="ok",
                n_rows=10,
                sum_abs=1.0,
                max_abs=9e-3,
                n_gt_atol=6,
            ),
            dict(
                match_key="a",
                model="mdl",
                feature="flat",
                status="ok",
                n_rows=10,
                sum_abs=0.0,
                max_abs=0.0,
                n_gt_atol=0,
            ),
            dict(
                match_key="c", model="mdl", feature="", status="empty", n_rows=0, sum_abs=0.0, max_abs=0.0, n_gt_atol=0
            ),
        ]
    )
    agg = _aggregate(combined)["mdl"]
    moved = agg["features"]["moved"]
    assert moved["n_rows"] == 20
    assert moved["mean_abs_delta"] == pytest.approx(2.0 / 20)
    assert moved["max_abs_delta"] == pytest.approx(9e-3)
    assert moved["frac_gt_atol"] == pytest.approx(10 / 20)
    assert "moved" in agg["moved_features"] and "flat" in agg["unmoved_features"]
    assert agg["max_abs_delta_over_features"] == pytest.approx(9e-3)
    assert agg["match_status_counts"]["empty"] == 1


def test_main_writes_provenance_stamped_metrics(tmp_path, monkeypatch):
    """End-to-end main() on a tc3-shaped cache: metrics.json carries models + provenance.

    Uses the real ghost-gk frame builder so at least one model produces `ok` rows; the point is the
    plumbing (for_each shard + reconcile + aggregate + provenance stamp), not the science.
    """
    from scripts.measure_f1b_feature_delta import main
    from silly_kicks.spadl import config as spc
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    cache = tmp_path / "cache"
    (cache / "shards" / "tok").mkdir(parents=True)
    (cache / "_actions").mkdir()
    (cache / "_home").mkdir()
    frames = pd.concat(
        [_make_ghost_gk_frames(frame_id=1, timestamp=1.0), _make_ghost_gk_frames(frame_id=2, timestamp=2.0)],
        ignore_index=True,
    )
    assert frames["x"].dtype == np.float64  # the driver refuses a float32-stored corpus
    frames.to_parquet(cache / "shards" / "tok" / "gradientsports__100.parquet")
    pd.DataFrame(
        {
            "game_id": ["100"],
            "period_id": [1],
            "team_id": [2],
            "time_seconds": [1.0],
            "type_id": [spc.actiontype_id["pass"]],
            "result_id": [spc.result_id["success"]],
        }
    ).to_parquet(cache / "_actions" / "gradientsports__100.parquet")
    (cache / "_home" / "gradientsports__100.json").write_text(json.dumps({"home_team_id": 1}))

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["measure_f1b_feature_delta.py", "--data-dir", str(cache), "--out", str(out), "--allow-dirty"]
    )
    main()

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["atol"] == _ATOL
    assert set(metrics["models"]) == {"xshot", "xcross", "ghost_gk", "ghost_outfield", "gk_completion", "receiver"}
    assert isinstance(metrics["run_tree_dirty"], bool)
    assert metrics["run_commit"]


def test_main_refuses_a_float32_corpus(tmp_path, monkeypatch):
    """A float32-stored corpus cannot expose the storage rounding -- the driver must fail loud."""
    from scripts.measure_f1b_feature_delta import main
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    cache = tmp_path / "cache"
    (cache / "shards" / "tok").mkdir(parents=True)
    frames = _make_ghost_gk_frames(frame_id=1, timestamp=1.0)
    frames[["x", "y"]] = frames[["x", "y"]].astype("float32")
    frames.to_parquet(cache / "shards" / "tok" / "gradientsports__100.parquet")

    out = tmp_path / "out"
    monkeypatch.setattr(
        sys, "argv", ["measure_f1b_feature_delta.py", "--data-dir", str(cache), "--out", str(out), "--allow-dirty"]
    )
    with pytest.raises(SystemExit, match="float64"):
        main()
