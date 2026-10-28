"""HFU plan R3-8 S3 item 15 (b): the ghost_outfield `coherence` stage (row 11; owner ruling 2026-10-31, NUMPY + split).

`fit` used to predict every training row after its fit and compute the coherence there, on one thread. The
`coherence` stage predicts 8 equal row chunks with the SAVED model; its reduce runs `ghost_rearguard_coherence` on
the chunks in key order. The `2474b73` `fit` coherence lines are the oracle below (verbatim, on the fitted model).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from scripts._stages import _encode, int_key
from scripts._train_guard import sklearn_supports_training

pytestmark = pytest.mark.skipif(
    not sklearn_supports_training(), reason="bundled-weights fit needs scikit-learn in [1.9, 2)"
)


def _oracle_coherence(final, data: pd.DataFrame) -> dict:
    """The `2474b73` `_fit_work` coherence lines, verbatim (``final`` is the fitted model)."""
    from silly_kicks.tracking._ghost_outfield import ghost_rearguard_coherence

    # Coherence (reported, not gated -- spec 9): served ghost positions on the training rows.
    served = data[["game_id", "period_id", "frame_id", "team_id", "slot_index"]].copy()
    fpred = final.predict_mean(data)
    served["ghost_gr_x"] = fpred[:, 0]
    served["ghost_gr_y"] = fpred[:, 1]
    return json.loads(json.dumps(ghost_rearguard_coherence(served)))


def _table(n_frames: int = 61, slots: int = 4) -> pd.DataFrame:
    from tests.scripts.hf_jobs.fixtures import make_synthetic_ghost_outfield_table

    t = make_synthetic_ghost_outfield_table(n_samples=n_frames * slots, seed=3)
    frame = np.repeat(np.arange(n_frames), slots)
    t["game_id"] = [f"g{f % 5}" for f in frame]
    t["period_id"] = 1 + (frame % 2)
    t["frame_id"] = frame
    t["team_id"] = np.where(frame % 3 == 0, 101, 202)
    t["slot_index"] = np.tile(np.arange(slots), n_frames)
    return t


def test_the_coherence_chunks_reproduce_the_whole_frame_coherence(tmp_path):
    """Each chunk goes through the store's own encoding (parquet) before the reduce, as in a job; the saved model
    predicts the chunks; the coherence equals the oracle on the fitted model, byte for byte as JSON."""
    import scripts.train_ghost_outfield as T
    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel

    data = _table()
    final = GhostOutfieldModel(n_estimators=8, max_depth=3)._fit_extracted(data)
    final.save(tmp_path / "m")
    loaded = GhostOutfieldModel.load(tmp_path / "m")
    results = {}
    for i in range(T.COHERENCE_CHUNKS):
        suffix, raw, _files = _encode(T.coherence_chunk(loaded, data, i))
        assert suffix == ".parquet"
        results[int_key(i)] = pd.read_parquet(__import__("io").BytesIO(raw))
    assert sum(len(r) for r in results.values()) == len(data)  # every row exactly once
    new = json.loads(json.dumps(T.coherence_of_chunks(results)))
    assert json.dumps(new) == json.dumps(_oracle_coherence(final, data))


def test_the_pipeline_runs_the_coherence_as_its_own_stage():
    import scripts.train_ghost_outfield as T

    no_ctx: Any = None
    st = T.PIPELINE.stage("coherence")
    assert st.units(no_ctx) == [int_key(i) for i in range(8)] and set(st.needs) == {"extract", "fit"}
    assert "coherence" in T.PIPELINE.stage("assemble").needs
    assert "ghost_rearguard_coherence" not in __import__("inspect").getsource(T._fit_work)


@pytest.mark.slow  # slow: one serial toy-corpus run of the whole ghost_outfield pipeline (ADR-023)
def test_the_assembled_coherence_equals_the_fit_time_coherence_on_the_toy_corpus(tmp_path):
    """The new pipeline's `metrics.json` coherence equals the `2474b73` fit-time lines on the same rows and model;
    the bundle no longer carries a `coherence.json` (it never shipped one)."""
    import tempfile

    import scripts.train_ghost_outfield as T
    from scripts._provenance import code_identity, env_provenance, git_provenance
    from scripts._stages import LocalStore, build_parser, make_context, materialize_fileset
    from scripts._toy_corpora import build_outfield_corpus
    from silly_kicks.tracking._ghost_outfield import GhostOutfieldModel
    from tests.scripts._parity import run_serial

    args = ["--dataset", "toyof", "--variant", "default", "--n-estimators", "10", "--max-depth", "3", "--cv-folds", "4"]
    data_root = tmp_path / "data"
    build_outfield_corpus(data_root / "toyof")
    run_serial(T.__name__, tmp_path / "s", data_root, args, tmp_path / "o")
    metrics = json.loads((tmp_path / "o" / "metrics.json").read_text(encoding="utf-8"))
    assert {p.name for p in (tmp_path / "o").iterdir()} == {"model.npz", "metadata.json", "SHA256SUMS", "metrics.json"}

    plan_args = ["--plan", "--store", str(tmp_path / "s"), "--data-root", str(data_root), *args]
    ns = build_parser(T.PIPELINE).parse_args(plan_args)
    git = git_provenance()
    ctx = make_context(
        T.PIPELINE,
        {p.name: getattr(ns, p.name) for p in T.PIPELINE.params},
        {**git, "env": env_provenance(), "code_identity": code_identity(git)},
        store=LocalStore(tmp_path / "s"),
        data_root=data_root,
    )
    with tempfile.TemporaryDirectory() as tmp:
        model = GhostOutfieldModel.load(materialize_fileset(ctx, "fit", "_output", Path(tmp) / "fit"))
    assert json.dumps(metrics["coherence"]) == json.dumps(_oracle_coherence(model, T._data(ctx)))
