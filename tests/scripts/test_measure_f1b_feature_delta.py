"""Unit + end-to-end tests for scripts/measure_f1b_feature_delta.py (ADR-106 T10).

The driver's SCIENCE (real per-feature float32 deltas) is validated on the owner corpus; here we
pin the pure plumbing -- the two-arm cast, the type_name restore, the KEY-ALIGNED per-feature delta
math + the selection-instability count, the status paths, the corpus rollup, and that main() writes
a provenance-stamped metrics.json. The key-alignment (vs the old positional row-for-row) is the fix
for the coord-selection-sensitive models; `test_delta_rows_key_alignment_not_positional` pins it.
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
    """An adapter returning `seq[i]` on its i-th call (arm f64 then f32); each item is (feats, keys)|None."""
    calls = {"n": 0}

    def adapter(_frames, _actions, _home):
        i = calls["n"]
        calls["n"] += 1
        return seq[i]

    return adapter


def _ok_rows(rows):
    return {r["feature"]: r for r in rows if r["status"] == "ok"}


def _selection_row(rows):
    sel = [r for r in rows if r["status"] == "selection"]
    assert len(sel) == 1
    return sel[0]


def test_delta_rows_computes_per_feature_delta():
    k = pd.DataFrame({"k": [1, 2, 3]})
    f64 = pd.DataFrame({"f": [1.0, 2.0, 3.0], "g": [0.0, 0.0, 0.0]})
    f32 = pd.DataFrame({"f": [1.0, 2.0 + 1e-3, 3.0], "g": [0.0, 0.0, 0.0]})
    rows = _delta_rows("m", "mdl", _stub_adapter([(f64, k), (f32, k)]), None, None, None, None)
    by = _ok_rows(rows)
    assert by["f"]["n_rows"] == 3
    assert by["f"]["max_abs"] == pytest.approx(1e-3)
    assert by["f"]["n_gt_atol"] == 1  # only row 1 exceeds atol
    assert by["g"]["max_abs"] == 0.0 and by["g"]["n_gt_atol"] == 0
    sel = _selection_row(rows)
    assert sel["n_rows"] == 3 and sel["n_only_f64"] == 0 and sel["n_only_f32"] == 0  # identical key sets


def test_delta_rows_key_alignment_not_positional():
    """The two arms select overlapping-but-different key SETS; the SAME key carries the SAME value.

    Key-aligned: the common keys {2,3} have delta 0, and one row is unique to each arm. A POSITIONAL
    row-for-row comparison (the pre-fix behaviour) would subtract different frames and report max_abs=10
    -- so this assertion fails on the old code and passes on the key-aligned code.
    """
    f64 = pd.DataFrame({"f": [10.0, 20.0, 30.0]})
    k64 = pd.DataFrame({"k": [1, 2, 3]})
    f32 = pd.DataFrame({"f": [20.0, 30.0, 40.0]})
    k32 = pd.DataFrame({"k": [2, 3, 4]})
    rows = _delta_rows("m", "mdl", _stub_adapter([(f64, k64), (f32, k32)]), None, None, None, None)
    by = _ok_rows(rows)
    assert by["f"]["max_abs"] == 0.0  # key-aligned: 2->20==20, 3->30==30 (positional would be 10.0)
    assert by["f"]["n_rows"] == 2  # only the 2 common keys are compared
    sel = _selection_row(rows)
    assert sel["n_only_f64"] == 1 and sel["n_only_f32"] == 1  # key 1 only in f64, key 4 only in f32
    assert sel["n_rows"] == 2


def test_delta_rows_status_paths():
    empty = _delta_rows("m", "mdl", _stub_adapter([None, None]), None, None, None, None)
    assert len(empty) == 1 and empty[0]["status"] == "empty"

    # A non-unique key makes the inner join ambiguous -> fail honestly (never a silent wrong match).
    kdup = pd.DataFrame({"k": [1, 1]})
    fdup = pd.DataFrame({"f": [1.0, 2.0]})
    dup = _delta_rows("m", "mdl", _stub_adapter([(fdup, kdup), (fdup, kdup)]), None, None, None, None)
    assert dup[0]["status"] == "dup_keys"

    # Disjoint key sets -> no common keys.
    none_common = _delta_rows(
        "m",
        "mdl",
        _stub_adapter(
            [
                (pd.DataFrame({"f": [1.0]}), pd.DataFrame({"k": [1]})),
                (pd.DataFrame({"f": [1.0]}), pd.DataFrame({"k": [2]})),
            ]
        ),
        None,
        None,
        None,
        None,
    )
    assert none_common[0]["status"] == "no_common_keys"

    def boom(_f, _a, _h):
        raise ValueError("nope")

    err = _delta_rows("m", "mdl", boom, None, None, None, None)
    assert err[0]["status"].startswith("error:ValueError")


def test_delta_rows_ignores_nonfinite_pairs():
    k = pd.DataFrame({"k": [1, 2, 3]})
    f64 = pd.DataFrame({"f": [1.0, np.nan, 3.0]})
    f32 = pd.DataFrame({"f": [1.0 + 2e-3, np.nan, 3.0]})
    rows = _delta_rows("m", "mdl", _stub_adapter([(f64, k), (f32, k)]), None, None, None, None)
    row = _ok_rows(rows)["f"]
    assert row["n_rows"] == 2  # the NaN pair is dropped
    assert row["max_abs"] == pytest.approx(2e-3)


def test_aggregate_rolls_up_counts_and_classifies():
    combined = pd.DataFrame(
        [
            dict(
                match_key="a",
                model="mdl",
                feature="moved",
                status="ok",
                n_rows=10,
                sum_abs=1.0,
                max_abs=5e-3,
                n_gt_atol=4,
                n_only_f64=0,
                n_only_f32=0,
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
                n_only_f64=0,
                n_only_f32=0,
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
                n_only_f64=0,
                n_only_f32=0,
            ),
            dict(
                match_key="a",
                model="mdl",
                feature="",
                status="selection",
                n_rows=10,
                sum_abs=0.0,
                max_abs=0.0,
                n_gt_atol=0,
                n_only_f64=1,
                n_only_f32=2,
            ),
            dict(
                match_key="b",
                model="mdl",
                feature="",
                status="selection",
                n_rows=10,
                sum_abs=0.0,
                max_abs=0.0,
                n_gt_atol=0,
                n_only_f64=0,
                n_only_f32=1,
            ),
            dict(
                match_key="c",
                model="mdl",
                feature="",
                status="empty",
                n_rows=0,
                sum_abs=0.0,
                max_abs=0.0,
                n_gt_atol=0,
                n_only_f64=0,
                n_only_f32=0,
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
    si = agg["selection_instability"]
    assert si["n_only_f64"] == 1 and si["n_only_f32"] == 3 and si["n_common"] == 20
    assert si["frac"] == pytest.approx(4 / 24)  # (1 + 3) / (1 + 3 + 20)


def test_prepare_xshot_return_keys_is_strictly_additive():
    """PROVENANCE GUARD (coordinator constraint): return_keys must not touch the training path.

    The default call returns the EXACT 3-tuple it always has (same feats/labels/groups); return_keys=True
    appends a 4th (game_id, period_id, frame_id) frame row-aligned to feats. The retrain anchors commit-1
    on the strength of this: weights are byte-identical whether trained before or after this change.
    """
    from silly_kicks.tracking._xshot_occurrence import prepare_xshot_training_data
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    frames = pd.concat(
        [_make_ghost_gk_frames(frame_id=1, timestamp=1.0), _make_ghost_gk_frames(frame_id=2, timestamp=2.0)],
        ignore_index=True,
    )
    if "ball_state" not in frames.columns:
        frames["ball_state"] = "alive"
    shots = pd.DataFrame(columns=["game_id", "period_id", "team_id", "time_seconds", "type_name"])

    default = prepare_xshot_training_data(frames, shots, home_team_id=1, feature_set="faithful")
    assert len(default) == 3  # unchanged 3-tuple

    keyed = prepare_xshot_training_data(frames, shots, home_team_id=1, feature_set="faithful", return_keys=True)
    assert len(keyed) == 4
    # The 3-tuple is byte-identical to the default call (the additive param cannot perturb it).
    assert keyed[0].equals(default[0])
    assert np.array_equal(keyed[1], default[1])
    assert np.array_equal(np.asarray(keyed[2], dtype=object), np.asarray(default[2], dtype=object))
    # The keys frame is row-aligned to features and carries the selected-frame identity.
    keys = keyed[3]
    assert list(keys.columns) == ["game_id", "period_id", "frame_id"]
    assert len(keys) == len(keyed[0])


def test_prepare_xcross_default_return_is_three_tuple():
    """xcross exposes keys via its existing return_meta=True; the driver uses that, so the DEFAULT
    (return_meta=False) training path stays the exact 3-tuple. Cheap provenance guard."""
    from silly_kicks.tracking._xcross_attempt import prepare_xcross_training_data
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    frames = _make_ghost_gk_frames(frame_id=1, timestamp=1.0)
    if "ball_state" not in frames.columns:
        frames["ball_state"] = "alive"
    actions = pd.DataFrame(columns=["game_id", "period_id", "team_id", "time_seconds", "type_id"])
    out = prepare_xcross_training_data(frames, actions, home_team_id=1, feature_set="faithful")
    assert len(out) == 3


def test_main_writes_provenance_stamped_metrics(tmp_path, monkeypatch):
    """End-to-end main() on a tc3-shaped cache: metrics.json carries models + provenance +
    the selection_instability block.

    Uses the real ghost-gk frame builder so at least one model produces `ok` rows; the point is the
    plumbing (for_each shard + reconcile + key-aligned aggregate + provenance stamp), not the science.
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
    # every model carries the selection-instability block (the key-alignment fix's reported quantity)
    for m in metrics["models"].values():
        assert set(m["selection_instability"]) == {"n_only_f64", "n_only_f32", "n_common", "frac"}


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


def _write_tc3_match(cache, gid: int, *, frame0: int) -> None:
    from silly_kicks.spadl import config as spc
    from tests.tracking.test_ghost_gk import _make_ghost_gk_frames

    frames = pd.concat(
        [
            _make_ghost_gk_frames(frame_id=frame0, timestamp=float(frame0)),
            _make_ghost_gk_frames(frame_id=frame0 + 1, timestamp=float(frame0 + 1)),
        ],
        ignore_index=True,
    )
    frames.to_parquet(cache / "shards" / "tok" / f"gradientsports__{gid}.parquet")
    pd.DataFrame(
        {
            "game_id": [str(gid)],
            "period_id": [1],
            "team_id": [2],
            "time_seconds": [float(frame0)],
            "type_id": [spc.actiontype_id["pass"]],
            "result_id": [spc.result_id["success"]],
        }
    ).to_parquet(cache / "_actions" / f"gradientsports__{gid}.parquet")
    (cache / "_home" / f"gradientsports__{gid}.json").write_text(json.dumps({"home_team_id": 1}))


def _tc3_cache(tmp_path):
    cache = tmp_path / "cache"
    (cache / "shards" / "tok").mkdir(parents=True)
    (cache / "_actions").mkdir()
    (cache / "_home").mkdir()
    _write_tc3_match(cache, 100, frame0=1)
    _write_tc3_match(cache, 101, frame0=3)
    return cache


def _t10(monkeypatch, *args):
    from scripts.measure_f1b_feature_delta import main

    monkeypatch.setattr(sys, "argv", ["measure_f1b_feature_delta.py", *map(str, args)])
    main()


_KEYS = ["shards__tok__gradientsports__100", "shards__tok__gradientsports__101"]


def _worker(monkeypatch, tmp_path, cache, out, i, keys):
    kj = tmp_path / f"k{i}.json"
    kj.write_text(json.dumps(keys))
    _t10(
        monkeypatch,
        "--data-dir",
        cache,
        "--out",
        out,
        "--shards-only",
        "--worker-tag",
        f"w{i}",
        "--match-keys-json",
        kj,
        "--allow-dirty",
    )


def _assert_same_artifact(serial, sharded):
    s = json.loads((serial / "metrics.json").read_text())
    p = json.loads((sharded / "metrics.json").read_text())
    assert p["models"] == s["models"]
    assert p["n_matches"] == s["n_matches"] == 2 and p["n_accounted"] == 2
    assert p["generation"] == s["generation"] and p["run_commit"] == s["run_commit"]
    a = pd.read_parquet(serial / "f1b_feature_delta.parquet")
    b = pd.read_parquet(sharded / "f1b_feature_delta.parquet")
    cols = list(a.columns)
    pd.testing.assert_frame_equal(
        a.sort_values(cols).reset_index(drop=True), b[cols].sort_values(cols).reset_index(drop=True)
    )


def test_sharded_run_reduces_to_the_serial_artifact(tmp_path, monkeypatch, capsys):
    cache = _tc3_cache(tmp_path)
    serial, sharded = tmp_path / "serial", tmp_path / "sharded"
    _t10(monkeypatch, "--data-dir", cache, "--out", serial, "--allow-dirty")
    capsys.readouterr()
    _t10(monkeypatch, "--data-dir", cache, "--list-match-keys")
    assert json.loads(capsys.readouterr().out) == _KEYS
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS[:1])
    _worker(monkeypatch, tmp_path, cache, sharded, 1, _KEYS[1:])
    assert not (sharded / "metrics.json").exists()  # a worker never writes the corpus artifact
    _t10(monkeypatch, "--data-dir", cache, "--out", sharded, "--reduce-only", "--allow-dirty")
    _assert_same_artifact(serial, sharded)


def test_a_worker_killed_before_its_manifest_still_reduces(tmp_path, monkeypatch):
    """Killed after its last shard but before writing manifest_<tag>.json: the commit-keyed generation
    still attributes the shards, so the reduce succeeds and matches serial (CCC-PLAN-11)."""
    cache = _tc3_cache(tmp_path)
    serial, sharded = tmp_path / "serial", tmp_path / "sharded"
    _t10(monkeypatch, "--data-dir", cache, "--out", serial, "--allow-dirty")
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS)
    for mf in (sharded / "_shards").glob("*/manifest_*.json"):
        mf.unlink()  # simulate the kill
    _t10(monkeypatch, "--data-dir", cache, "--out", sharded, "--reduce-only", "--allow-dirty")
    _assert_same_artifact(serial, sharded)


def test_a_worker_resumed_with_the_same_tag_reduces_identically(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    serial, sharded = tmp_path / "serial", tmp_path / "sharded"
    _t10(monkeypatch, "--data-dir", cache, "--out", serial, "--allow-dirty")
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS[:1])  # "killed" after one item
    _worker(monkeypatch, tmp_path, cache, sharded, 0, _KEYS)  # relaunched: resumes, attempts only the rest
    _t10(monkeypatch, "--data-dir", cache, "--out", sharded, "--reduce-only", "--allow-dirty")
    _assert_same_artifact(serial, sharded)


def test_reduce_refuses_an_unfinished_corpus(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    out = tmp_path / "o"
    _worker(monkeypatch, tmp_path, cache, out, 0, _KEYS[:1])
    with pytest.raises(SystemExit, match="have no shard"):
        _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")


def test_reduce_refuses_a_manifest_from_another_commit(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    out = tmp_path / "o"
    _worker(monkeypatch, tmp_path, cache, out, 0, _KEYS)
    (mf,) = list((out / "_shards").glob("*/manifest_w0.json"))
    m = json.loads(mf.read_text())
    m["run_commit"] = "deadbeef"
    mf.write_text(json.dumps(m))
    with pytest.raises(SystemExit, match="another commit"):
        _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")


def test_the_generation_is_keyed_on_the_commit(tmp_path, monkeypatch):
    """B r4 CCC-PLAN-31: a worker at commit A that died before its manifest is refused by a reduce at commit
    B. No manifest is left, so the commit-keyed GENERATION is what refuses (the CCC-SPEC-04 mechanism); the
    same reduce at commit A succeeds, so the commit is the only difference."""
    import scripts.measure_f1b_feature_delta as t10

    def _at(commit):
        monkeypatch.setattr(
            t10, "git_provenance", lambda: {"commit": commit * 40, "dirty": False, "tree_state": "clean"}
        )

    cache = _tc3_cache(tmp_path)
    out = tmp_path / "o"
    _at("a")
    _worker(monkeypatch, tmp_path, cache, out, 0, _KEYS)
    for mf in (out / "_shards").glob("*/manifest_*.json"):
        mf.unlink()  # killed before its manifest
    _at("b")
    with pytest.raises(SystemExit, match="expected exactly the generation"):
        _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")
    _at("a")
    _t10(monkeypatch, "--data-dir", cache, "--out", out, "--reduce-only", "--allow-dirty")
    assert json.loads((out / "metrics.json").read_text())["run_commit"] == "a" * 40


def test_shards_only_requires_a_worker_tag(tmp_path, monkeypatch):
    """Must FAIL before the change for the right reason: assert the message, not just SystemExit."""
    cache = _tc3_cache(tmp_path)
    with pytest.raises(SystemExit):
        _t10(monkeypatch, "--data-dir", cache, "--out", tmp_path / "o", "--shards-only", "--allow-dirty")
    # argparse prints its error; the plan's message is the discriminator
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), pytest.raises(SystemExit):
        _t10(monkeypatch, "--data-dir", cache, "--out", tmp_path / "o", "--shards-only", "--allow-dirty")
    assert "--shards-only needs a unique --worker-tag" in buf.getvalue()


def test_unknown_match_key_is_refused(tmp_path, monkeypatch):
    cache = _tc3_cache(tmp_path)
    kj = tmp_path / "k.json"
    kj.write_text(json.dumps(["nope"]))
    with pytest.raises(SystemExit, match="absent from --data-dir"):
        _t10(
            monkeypatch,
            "--data-dir",
            cache,
            "--out",
            tmp_path / "o",
            "--shards-only",
            "--worker-tag",
            "w0",
            "--match-keys-json",
            kj,
            "--allow-dirty",
        )
