"""ADR-128 W1N1-vs-W2N2 parity for `train_match_outcome_dependence` (public source, per-match fan-out; HFU spec §5)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("statsbombpy")

from scripts._toy_corpora import build_open_data
from tests.scripts._open_data_server import serve_open_data
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial

MODULE = "scripts.train_match_outcome_dependence"


@pytest.mark.slow  # slow: the same toy open-data corpus as the win_probability parity test (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    root = build_open_data(tmp_path / "od")
    comps = tmp_path / "competitions.json"
    comps.write_text("[[43, 106]]", encoding="utf-8")
    args = ["--competitions-json", str(comps)]
    with serve_open_data(root) as url:
        env = {**os.environ, "SILLY_KICKS_OPEN_DATA_BASE": url}
        run_serial(MODULE, tmp_path / "s1", tmp_path / "data", args, tmp_path / "o1", env=env)
        run_fanned(MODULE, tmp_path / "s2", tmp_path / "data", args, tmp_path / "o2", env=env)
        gens = {
            s["name"]: s["generation"]
            for s in plan(MODULE, tmp_path / "s2", tmp_path / "data", args, env=env)["stages"]
        }
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert set(artifact(tmp_path / "o1")) == {"model.json", "SHA256SUMS"}
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
