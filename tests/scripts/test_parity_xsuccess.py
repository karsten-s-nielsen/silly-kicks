"""ADR-128 W1N1-vs-W2N2 parity for `train_xsuccess` (public source; the HPO folds on the InnerPool; HFU spec
15.5a, 15.7): the fold-parallel study gives the serial result byte for byte."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("statsbombpy")
pytest.importorskip("xgboost")

from scripts._toy_corpora import build_open_data
from tests.scripts._open_data_server import serve_open_data
from tests.scripts._parity import (
    artifact,
    completion_out_of_order,
    inner_out_of_order,
    outputs,
    plan,
    run_fanned,
    run_serial,
)

MODULE = "scripts.train_xsuccess"
ARGS = ["--competition-id", "43", "--season-id", "106", "--hpo-trials", "3"]


@pytest.mark.slow  # slow: the toy open-data corpus plus a 3-trial study (ADR-023)
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
    assert {"model.json", "SHA256SUMS", "metadata.json", "metrics.json", "MODEL_CARD.md"} <= set(
        artifact(tmp_path / "o1")
    )
    assert completion_out_of_order(tmp_path / "s2", gens["extract"])
    assert inner_out_of_order(tmp_path / "s2", gens["hpo"])  # the folds really ran on the pool, out of order
    # The fit folds and final fit ran as InnerPool units (HFU spec 16.2). Their completion order depends on timing
    # (the final fit, submitted first and sorted last, often but not always finishes last), so an out-of-order
    # assertion would be flaky (owner 2026-11-09); the deterministic order proof is the reversed pool in
    # test_fit_map_equals_the_serial_fit.
    keys = _inner_keys(tmp_path / "s2", gens["fit"])  # n_splits = min(5, matches) folds, then the final fit
    assert len(keys) >= 3 and keys == [f"{i:04d}" for i in range(len(keys) - 1)] + ["final"]


def _inner_keys(store, generation: str) -> list[str]:
    """The sorted inner keys a unit of generation completed on its InnerPool (from its ledger)."""
    import json
    import pathlib

    recs = [json.loads(p.read_text()) for p in (pathlib.Path(store) / generation).glob("*.ledger.json")]
    return sorted(k for r in recs for k in (r.get("inner_completion") or []))
