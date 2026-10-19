"""ADR-128 W1N1-vs-W2N2 parity for `train_receiver_model` (owner_sb360 staged source; INV-5)."""

from __future__ import annotations

import pytest

from scripts._toy_corpora import build_receiver_corpus
from tests.scripts._parity import artifact, completion_out_of_order, outputs, plan, run_fanned, run_serial

MODULE = "scripts.train_receiver_model"
ARGS = [
    "--dataset",
    "toyrecv",
    "--provider",
    "statsbomb",
    "--feature-set",
    "public",
    "--min-passes",
    "1",
    "--corpus-visibility",
    "restricted",
    "--corpus-n-matches",
    "4",
]


@pytest.mark.slow  # slow: measured 188 s on the Windows dev box (ADR-023)
def test_w1n1_equals_w2n2(tmp_path):
    build_receiver_corpus(tmp_path / "data" / "toyrecv", n_games=4)
    data = tmp_path / "data"
    run_serial(MODULE, tmp_path / "s1", data, ARGS, tmp_path / "o1")
    run_fanned(MODULE, tmp_path / "s2", data, ARGS, tmp_path / "o2")
    assert outputs(tmp_path / "s1") == outputs(tmp_path / "s2")
    assert artifact(tmp_path / "o1") == artifact(tmp_path / "o2")
    assert set(artifact(tmp_path / "o1")) == {"model.json", "SHA256SUMS", "metrics.json"}
    gens = {s["name"]: s["generation"] for s in plan(MODULE, tmp_path / "s2", data, ARGS)["stages"]}
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
