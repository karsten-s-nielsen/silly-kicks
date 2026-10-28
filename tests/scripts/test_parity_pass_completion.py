"""ADR-128 W1N1-vs-W2N2 parity for `train_pass_completion` (public open-data source; HFU R3-10)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("statsbombpy")

from scripts._toy_corpora import build_open_data
from tests.scripts._open_data_server import serve_open_data
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial

MODULE = "scripts.train_pass_completion"
ARGS = ["--source", "open-data"]  # the campaign's build args: WC2022 (the toy tree's one competition)


@pytest.mark.slow  # slow: two full pipeline runs on the served toy tree (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    root = build_open_data(tmp_path / "od")
    with serve_open_data(root) as url:
        env = {**os.environ, "SILLY_KICKS_OPEN_DATA_BASE": url}
        run_serial(MODULE, tmp_path / "s1", tmp_path / "data", ARGS, tmp_path / "o1", env=env)
        run_fanned(MODULE, tmp_path / "s2", tmp_path / "data", ARGS, tmp_path / "o2", env=env)
        gens = {
            s["name"]: s["generation"]
            for s in plan(MODULE, tmp_path / "s2", tmp_path / "data", ARGS, env=env)["stages"]
        }
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert set(artifact(tmp_path / "o1")) == {
        "MODEL_CARD.md",
        "SHA256SUMS",
        "metrics.json",
        "model.json",
    }  # flat, ships metrics.json
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
