"""ADR-128 W1N1-vs-W2N2 parity for `train_win_probability` (public source, large-N fan-out)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("statsbombpy")

from scripts._toy_corpora import build_open_data
from tests.scripts._open_data_server import serve_open_data
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial

MODULE = "scripts.train_win_probability"
ARGS = ["--competition-id", "43", "--season-id", "106"]


@pytest.mark.slow  # slow: measured 140 s on the Windows dev box (ADR-023)
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
    assert set(artifact(tmp_path / "o1")) == {"model.json", "SHA256SUMS", "metrics.json", "MODEL_CARD.md"}
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
