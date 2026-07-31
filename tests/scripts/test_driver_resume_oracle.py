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
