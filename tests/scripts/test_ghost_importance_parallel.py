"""Decision P11: importance at W=2 is byte-identical to W=1 and really ran on >= 2 worker processes."""

from __future__ import annotations

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
    for name, inner in (("a", "1"), ("b", "2")):
        cli(
            MODULE,
            tmp_path / name,
            data,
            ARGS,
            "--run-all",
            "--inner-workers",
            inner,
            env=env if inner == "2" else None,
        )
    gen = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "a", data, ARGS)["stages"]}["importance"]
    a = (tmp_path / "a" / gen / "only.json").read_bytes()
    b = (tmp_path / "b" / gen / "only.json").read_bytes()
    assert a == b
    assert len(set(log.read_text().split())) >= 2
