"""The double-invocation oracle: run a driver twice, prove the second run does no work.

WHY THIS EXISTS. Before this file, no test on any of the three resumable drivers exercised the
resume branch. `test_build_layer2_spells.py` runs `main()` once and asserts the shard exists, never
re-running to reach `if shard.is_file(): continue`; `test_build_gkdv_arm_values.py`'s tests never
touch the shard loop; and `test_an_EMPTY_shard_is_distinct_from_an_ABSENT_one` tests the resume
check's PRECONDITION rather than the skip. So the safety net covered writes and aggregation and was
blind to resume -- while the migration changes shard paths.

The empty-shard round trip is included deliberately. Losing it is invisible except as a slow resume,
and it is currently pinned for only one of the three drivers.

Built BEFORE any driver is touched, so it characterises today's correct behaviour and the migration
has to preserve it. A test written after the change would only describe the change.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

import scripts.build_layer2_spells as layer2


@pytest.fixture()
def stub_layer2(monkeypatch):
    """Stub the corpus loader and the expensive builders; count entries into the real work.

    `build_layer2_spells.main()` imports these INSIDE the function body, so patching the source
    module is what the driver actually resolves at call time.
    """
    entered: list[str] = []

    import scripts._loader_pining as loader
    import silly_kicks.causal as causal
    import silly_kicks.causal._confounders as conf

    def _load(**_kw):
        return iter(
            [
                ("gradientsports", "m1", object(), object(), "5"),
                ("gradientsports", "barren", object(), object(), "5"),
            ]
        )

    def _build(frames, actions, **_kw):
        entered.append("build")
        # "barren" is the second item; return an EMPTY frame for it.
        return pd.DataFrame() if len(entered) == 2 else pd.DataFrame({"Z": [0, 1], "r": [1.0, 2.0]})

    monkeypatch.setattr(loader, "load_matches", _load)
    monkeypatch.setattr(causal, "build_opportunities", _build)
    monkeypatch.setattr(causal, "layer2_config", lambda *a, **k: object())
    monkeypatch.setattr(conf, "join_layer2_confounders", lambda sp, **k: sp)
    return entered


def _run(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_layer2_spells.py", "--out", str(tmp_path), "--allow-dirty"])
    layer2.main()


def test_layer2_second_run_does_NO_work(tmp_path, monkeypatch, stub_layer2):
    _run(tmp_path, monkeypatch)
    first = len(stub_layer2)
    assert first == 2, "the first run should have built both matches"

    _run(tmp_path, monkeypatch)

    assert len(stub_layer2) == first, "the second run re-entered the expensive builder"


def test_layer2_second_run_produces_an_IDENTICAL_table(tmp_path, monkeypatch, stub_layer2):
    _run(tmp_path, monkeypatch)
    before = (tmp_path / "layer2_spells.parquet").read_bytes()
    _run(tmp_path, monkeypatch)
    assert (tmp_path / "layer2_spells.parquet").read_bytes() == before


def test_layer2_a_BARREN_match_is_not_recomputed(tmp_path, monkeypatch, stub_layer2):
    """The empty-shard round trip. A match producing zero rows must leave a shard behind and be
    skipped on re-run -- otherwise every barren match recomputes forever, which is the exact cost
    this cycle exists to remove."""
    _run(tmp_path, monkeypatch)
    shards = list(tmp_path.rglob("*barren*.parquet"))
    assert shards, "the barren match left no shard, so a resume will recompute it"
    assert pd.read_parquet(shards[0]).empty

    _run(tmp_path, monkeypatch)

    assert len(stub_layer2) == 2, "the barren match was recomputed on resume"


def test_the_oracle_would_CATCH_a_broken_resume(tmp_path, monkeypatch, stub_layer2):
    """Non-vacuity: with the resume check disabled, the oracle must fail. Without this, a green
    oracle is indistinguishable from one that never exercised resume at all."""
    _run(tmp_path, monkeypatch)
    first = len(stub_layer2)
    for shard in tmp_path.rglob("*.parquet"):
        if shard.parent != tmp_path:
            shard.unlink()  # simulate a migration that lost resume

    _run(tmp_path, monkeypatch)

    assert len(stub_layer2) > first, "the oracle cannot detect a lost resume -- it is vacuous"


def _manifest(tmp_path) -> dict:
    import json

    return json.loads((tmp_path / "manifest_all.json").read_text(encoding="utf-8"))


def test_layer2_manifest_REPORTS_counters_it_could_not_replay(tmp_path, monkeypatch, stub_layer2):
    """Corpus totals in a worker manifest come from the per-item counters sidecars on a resume.
    When a sidecar is absent -- a pre-sidecar generation, or one truncated by a kill -- the
    manifest has to SAY the totals under-report the corpus, instead of printing a smaller number
    that reads as complete. `_partition.aggregate_manifests` sums these across workers, so one
    silent worker is enough to mis-state a corpus artifact.

    The only thing holding this up is the driver calling `res.manifest()`: a hand-written
    `manifest_fields(shard_dir, attempted=..., failed=...)` defaults `counters_unrecorded` to 0 and
    is indistinguishable, in the artifact, from a run that replayed everything.
    """
    _run(tmp_path, monkeypatch)
    clean = _manifest(tmp_path)
    assert clean["n_counters_unrecorded"] == 0
    assert clean["n_matches"] == 2, "the first pass attempted both, so its totals are complete"

    for sidecar in tmp_path.rglob("*.counters.json"):
        sidecar.unlink()
    _run(tmp_path, monkeypatch)

    resumed = _manifest(tmp_path)
    assert resumed["n_counters_unrecorded"] == 2, "the manifest hid that it could replay neither"
    # And the totals it does print are honestly ABSENT rather than a smaller-looking corpus.
    assert resumed.get("n_matches", 0) == 0


def test_the_sidecar_replay_is_what_keeps_a_resumed_manifest_complete(tmp_path, monkeypatch, stub_layer2):
    """The other side. Leave the sidecars in place and the SAME resumed pass -- which attempts
    nothing -- still reports the full corpus. Without this half, the test above passes on a build
    that lost counter replay entirely."""
    _run(tmp_path, monkeypatch)
    _run(tmp_path, monkeypatch)

    resumed = _manifest(tmp_path)
    assert resumed["n_counters_unrecorded"] == 0
    assert resumed["n_matches"] == 2, "a fully resumed pass lost the corpus totals"
    assert resumed["n_attempted"] == 0, "and it must still declare it BUILT nothing this pass"
