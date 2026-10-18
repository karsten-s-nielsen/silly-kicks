"""ADR-123 receiver extraction fan-out: a --shard-index worker extracts its disjoint ref subset and EXITS
before the reconcile/fit; the partition is a disjoint cover. Offline (monkeypatched corpus source), so it
runs on any env -- the real fit is covered by the trainer's own corpus smokes + the DGX e2e."""

from __future__ import annotations

import sys

import pandas as pd

import scripts.train_receiver_model as trm


class _Ref:
    def __init__(self, mid):
        self.match_id = mid


def _fake_source(n=4):
    refs = [_Ref(f"m{i}") for i in range(n)]

    def load(ref):
        actions = pd.DataFrame({"game_id": ["g"], "action_id": [0]})
        frames = pd.DataFrame({"x": [1.0]})
        return (ref, actions, frames)

    return refs, load


def test_shard_index_extracts_a_disjoint_subset_and_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(trm, "_corpus_source", lambda provider, cache_dir, match_ids=None: _fake_source(4))
    monkeypatch.setattr(
        trm,
        "extract_candidate_rows",
        lambda a, f, *, feature_set, labeling_strategy: pd.DataFrame(
            {"game_id": ["g"], "action_id": [0], "label": [1]}
        ),
    )
    monkeypatch.setattr(trm, "_coverage_counters", lambda item, frame: {})
    sr = tmp_path / "shards"
    rows, _cov = trm._extract_provider_rows(
        "statsbomb", "public", sr, None, tmp_path / "out.parquet", "all", shard_index="0/2"
    )
    # Extract-only worker returns EMPTY rows (the reduce reconciles + fits).
    assert len(rows) == 0
    # Wrote shards for refs m0, m2 only (0/2 == a disjoint cover's first half).
    written = sorted(p.name for p in sr.glob("**/*.parquet"))
    stems = " ".join(written)
    assert "m0" in stems and "m2" in stems
    assert "m1" not in stems and "m3" not in stems


def test_main_shard_index_exits_before_fit(tmp_path, monkeypatch):
    calls = {}

    def fake_extract(provider, fs, sr, cd, out, tag, match_ids=None, shard_index=None, data_dir=None):
        calls["shard_index"] = shard_index
        return pd.DataFrame(), {}

    monkeypatch.setattr(trm, "git_provenance", lambda: {"commit": "c", "dirty": True, "tree_state": "dirty"})
    monkeypatch.setattr(trm, "require_clean_tree", lambda prov, allow_dirty=False: prov)
    monkeypatch.setattr(trm, "_extract_provider_rows", fake_extract)
    out = tmp_path / "o"
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--out", str(out), "--shard-root", str(tmp_path / "s"), "--shard-index", "1/2", "--allow-dirty"],
    )
    trm.main()
    assert calls["shard_index"] == "1/2"
    assert not (out / "model").exists()  # exited before the fit + save


def test_data_dir_source_reads_flat_frames_and_actions_no_pining(tmp_path):
    # ADR-123 agnostic source: the STAGED flat shards + _actions (eval shape), yielding (match_id, actions,
    # frames) exactly like the pining _corpus_source -- no pining import, no manifest.
    gen = tmp_path / "shards" / "g1"
    gen.mkdir(parents=True)
    (tmp_path / "_actions").mkdir()
    pd.DataFrame({"game_id": ["M1"], "x": [1.0]}).to_parquet(gen / "sb__M1.parquet")
    pd.DataFrame({"type_id": [0], "action_id": [0]}).to_parquet(tmp_path / "_actions" / "sb__M1.parquet")
    refs, load = trm._data_dir_source(tmp_path)
    assert [r.match_id for r in refs] == ["sb__M1"]
    mid, actions, frames = load(refs[0])
    assert mid == "sb__M1" and len(actions) == 1 and len(frames) == 1


def test_main_shard_index_rejects_pool_provider(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(trm, "git_provenance", lambda: {"commit": "c", "dirty": True, "tree_state": "dirty"})
    monkeypatch.setattr(trm, "require_clean_tree", lambda prov, allow_dirty=False: prov)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--out",
            str(tmp_path / "o"),
            "--shard-root",
            str(tmp_path / "s"),
            "--shard-index",
            "0/2",
            "--pool-provider",
            "gradientsports",
            "--allow-dirty",
        ],
    )
    with pytest.raises(SystemExit, match="does not support --pool-provider"):
        trm.main()
