"""ADR-128 spec 5.6: the one-unit adapter stage runs an unmigrated trainer's CLI unchanged as ONE unit."""

from __future__ import annotations

import json
import os

from tests.scripts._parity import cli, plan

MODULE = "scripts.hf_jobs.adapter"
ARGS = [
    "--target",
    "tests.scripts._toy_legacy_trainer",
    "--target-args",
    '["--seed", "3"]',
    "--fresh-subdir",
    "weights",
]


def _target_out(store, data):
    gen = {s["name"]: s["generation"] for s in plan(MODULE, store, data, ARGS)["stages"]}["run"]
    return store / gen / "target_out"


def test_the_run_output_is_the_targets_artifact_under_the_deterministic_env(tmp_path):
    store, data = tmp_path / "store", tmp_path / "data"
    data.mkdir()
    cli(MODULE, store, data, ARGS, "--run-all", "--out", str(tmp_path / "out"))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["model.json"]  # weights/* only
    model = json.loads((tmp_path / "out" / "model.json").read_text(encoding="utf-8"))
    assert model == {"sum": 9, "omp_threads": "1"}  # seed 3 -> 3 + 6; INV-2: the target ran single-threaded


def test_a_rerun_is_a_no_op(tmp_path):
    store, data = tmp_path / "store", tmp_path / "data"
    data.mkdir()
    for _ in range(2):
        cli(MODULE, store, data, ARGS, "--run-all")
    assert (_target_out(store, data) / "runs.log").read_text(encoding="utf-8").splitlines() == ["run"]


def test_a_killed_attempt_resumes_through_the_targets_own_shards(tmp_path):
    """The kill is an ENVIRONMENT fact (never a parameter), so the second attempt has the same token: it
    reuses the persistent --out, and the target skips the shard it already wrote."""
    store, data = tmp_path / "store", tmp_path / "data"
    data.mkdir()
    killed = cli(MODULE, store, data, ARGS, "--run-all", check=False, env={**os.environ, "TOY_DIE_AFTER_FIRST": "1"})
    assert killed.returncode != 0
    out = _target_out(store, data)
    assert (out / "writes.log").read_text(encoding="utf-8").splitlines() == ["m0"]
    cli(MODULE, store, data, ARGS, "--run-all", "--out", str(tmp_path / "out"))
    assert (out / "runs.log").read_text(encoding="utf-8").splitlines() == ["run", "run"]
    assert (out / "writes.log").read_text(encoding="utf-8").splitlines() == ["m0", "m1"], "m0 was recomputed"
    assert json.loads((tmp_path / "out" / "model.json").read_text(encoding="utf-8"))["sum"] == 9
