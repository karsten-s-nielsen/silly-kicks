"""Spec r3 D10 (supersedes decision P11): the ghost_gk importance stage runs one unit per column, so its table at
W=2 is byte-identical to W=1 and its units really ran in >= 2 worker processes."""

from __future__ import annotations

import json
import os

import pytest

from scripts._toy_corpora import build_toy_for
from scripts._train_guard import sklearn_supports_training
from tests.scripts._parity import cli, plan

pytestmark = [pytest.mark.skipif(not sklearn_supports_training(), reason="needs sklearn>=1.9")]
MODULE = "scripts.train_ghost_gk"
ARGS = [
    "--dataset",
    "toygk",
    "--variant",
    "default",
    "--n-estimators",
    "10",
    "--max-depth",
    "3",
    "--cv-folds",
    "2",
    "--perm-importance-sample",
    "200",
]


@pytest.mark.slow  # slow: measured 144 s on the Windows dev box (ADR-023)
def test_importance_w1_equals_w2_and_uses_two_processes(tmp_path):
    data = build_toy_for(MODULE, tmp_path / "data")
    log = tmp_path / "pids.log"
    env = {**os.environ, "SILLY_KICKS_IMPORTANCE_PID_LOG": str(log)}
    for name, workers in (("a", "1"), ("b", "2")):
        cli(
            MODULE,
            tmp_path / name,
            data,
            ARGS,
            "--run-all",
            "--workers",
            workers,
            env=env if workers == "2" else None,
        )
    gen = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "a", data, ARGS)["stages"]}["importance"]
    a = (tmp_path / "a" / gen / "_output.json").read_bytes()
    b = (tmp_path / "b" / gen / "_output.json").read_bytes()
    assert a == b
    assert json.loads(a)["skipped"] is False and json.loads(a)["top"]  # non-vacuity: a real table
    assert len(set(log.read_text().split())) >= 2
