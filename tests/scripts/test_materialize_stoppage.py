"""TF-46 Task 2 (materialize re-stage): the tc3-frames-4 schema-token bump + the silent-shrink guard.

The staging run attaches stoppage per match (Task 1 pins the loader seam). Task 2's own guarantees are
(a) the staged shards declare the bumped schema token, and (b) a staging run REFUSES a silently-shrunk
corpus. ``for_each`` records a raising item (e.g. the stoppage-column ``AssertionError`` in ``_work``) as a
FAILURE and carries on, so without the guard a dropped match would still exit 0 (r4 §3 path 1). The guard
is ``_assert_full_corpus``; these pin it from both sides (B02)."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from scripts._driver import CorpusPassResult
from scripts.materialize_tc3_frames import _SHARD_SCHEMA_VERSION, _assert_full_corpus


def _result(keys: tuple[str, ...], *, failures: dict | None = None, excluded: int = 0, exclusions: dict | None = None):
    """A minimal ``CorpusPassResult`` for the guard; ``shard_keys`` = keys minus failures minus exclusions."""
    failures = failures or {}
    exclusions = exclusions or {}
    return CorpusPassResult(
        shard_dir=pathlib.Path("gen"),
        attempted=len(keys),
        skipped=0,
        failed=len(failures),
        failures=failures,
        counters={},
        keys=keys,
        excluded=excluded,
        exclusions=exclusions,
    )


def test_schema_token_bumped_to_v4():
    # tc3-frames-4: the staged shard MUST carry the stoppage columns; the token change forces a new generation.
    assert _SHARD_SCHEMA_VERSION == "tc3-frames-4"


def test_full_corpus_guard_raises_on_any_failure():
    # a non-consecutive attach failure drops a match; for_each still exits 0, so the guard must fail hard
    res = _result(("gradientsports__10502",), failures={"gradientsports__10503": "AssertionError: stoppage col"})
    with pytest.raises(SystemExit, match="FAILED and were dropped"):
        _assert_full_corpus(res, {"gradientsports": ["10502", "10503"]})


def test_full_corpus_guard_raises_on_short_roster():
    # roster asked for 2, only 1 shard present and no exclusion recorded -> the corpus silently shrank (§4.3)
    res = _result(("gradientsports__10502",))
    with pytest.raises(SystemExit, match="refusing to stage a short set"):
        _assert_full_corpus(res, {"gradientsports": ["10502", "10503"]})


def test_full_corpus_guard_passes_on_complete_roster():
    res = _result(("gradientsports__10502", "gradientsports__10503"))
    _assert_full_corpus(res, {"gradientsports": ["10502", "10503"]})  # no raise: every requested match staged


def test_full_corpus_guard_counts_a_declared_exclusion_as_accounted():
    # an EXCLUDED match (a deterministic MatchExcluded admission decision) is not a shrink: roster 2, 1 shard,
    # 1 excluded -> expected = 2 - 1 = 1, so the guard accepts.
    res = _result(
        ("gradientsports__10502", "gradientsports__10503"),
        excluded=1,
        exclusions={"gradientsports__10503": "MatchExcluded: geometry"},
    )
    _assert_full_corpus(res, {"gradientsports": ["10502", "10503"]})  # no raise


def _run_main(monkeypatch, tmp_path, *extra: str, for_each) -> None:
    """Drive ``materialize_tc3_frames.main()`` offline: no pining, the given ``for_each`` stub, a dirty-tree pass."""
    import sys

    import scripts._loader_pining as lp
    from scripts import materialize_tc3_frames as m

    monkeypatch.setattr(lp, "resolve_cache_dir", lambda p: p)
    monkeypatch.setattr(lp, "pining_source", lambda *a, **k: ([], lambda ref: ref))
    monkeypatch.setattr(m, "for_each", for_each)
    argv = ["materialize", "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "o")]
    monkeypatch.setattr(sys, "argv", [*argv, "--providers", "sportec", "--allow-dirty", *extra])
    m.main()


def test_main_wires_the_full_corpus_guard(monkeypatch, tmp_path):
    # B16: the helper is tested above, but deleting the `_assert_full_corpus(res, match_ids)` CALL in main()
    # must not pass silently. Drive main() with a for_each that reports a failure; main() must exit non-zero
    # THROUGH the guard. Removing the call lets main() proceed past the failure -> this stops raising.
    failing = _result(("gradientsports__1",), failures={"gradientsports__1": "AssertionError: stoppage col"})
    with pytest.raises(SystemExit, match="FAILED and were dropped"):
        _run_main(monkeypatch, tmp_path, for_each=lambda *a, **k: failing)


def _stoppage_frames() -> pd.DataFrame:
    """One flag-on (tc3-frames-4) frame set: the loader attached the stoppage columns."""
    return pd.DataFrame(
        {
            "source_provider": ["sportec", "sportec"],
            "game_id": ["g1", "g1"],
            "period_id": [1, 1],
            "frame_id": [1, 2],
            "x": [1.0, 2.0],
            "stoppage_state": ["live", "dead"],
            "stoppage_source": ["events_v3", "events_v3"],
        }
    )


def test_first_shard_presence_assertion(monkeypatch, tmp_path):
    # Plan Task 2 / spec section 7: the staged shard MUST carry the stoppage columns. Capture main()'s REAL
    # `_work` and feed it one shard with and one without them: the flag-on shard passes, the dropped column
    # refuses. Disabling the presence check in `_work` lets the second shard through -> this goes red.
    captured: dict = {}

    def capture(*a, **k):
        captured["work"] = k["work"]
        return _result(())

    _run_main(monkeypatch, tmp_path, for_each=capture)
    frames = _stoppage_frames()
    assert captured["work"](("sportec", "m1", None, frames, "h")) is frames
    with pytest.raises(AssertionError, match="missing stoppage column"):
        captured["work"](("sportec", "m2", None, frames.drop(columns="stoppage_source"), "h"))


def test_main_loads_the_preflight_reference_flag_on(monkeypatch, tmp_path):
    # SPEC-13 wiring: main() hands the pre-flight a loader with the stoppage flag ON (the write path's flag).
    # Switching that partial to flag-off makes this go red.
    from scripts import materialize_tc3_frames as m

    seen: dict = {}

    def fake_parity(path, *, cache_dir, list_match_refs, load_match):
        seen["keywords"] = dict(getattr(load_match, "keywords", {}))
        return "m1"

    monkeypatch.setattr(m, "preflight_reference_parity", fake_parity)
    ref = str(tmp_path / "ref.parquet")
    _run_main(monkeypatch, tmp_path, "--reference-parquet", ref, for_each=lambda *a, **k: _result(()))
    assert seen["keywords"] == {"stoppage": True}


def _preflight(tmp_path, *, stoppage: bool) -> str:
    """Run the REAL pre-flight against a flag-on reference, with a loader that attaches stoppage iff asked."""
    import types
    from functools import partial

    from scripts.materialize_tc3_frames import preflight_reference_parity

    frames = _stoppage_frames()
    ref = tmp_path / "sportec" / "m1" / "frames.parquet"
    ref.parent.mkdir(parents=True)
    frames.to_parquet(ref)  # the regenerated (flag-on) reference

    def load_match(_ref, *, events_only, cache_dir, stoppage=False):
        out = frames if stoppage else frames.drop(columns=["stoppage_state", "stoppage_source"])
        return types.SimpleNamespace(frames=out, match_id="m1")

    loader = partial(load_match, stoppage=True) if stoppage else load_match
    return preflight_reference_parity(ref, cache_dir=None, list_match_refs=lambda **k: ["ref"], load_match=loader)


def test_preflight_flag_on_vs_regenerated_reference_passes(tmp_path):
    assert _preflight(tmp_path, stoppage=True) == "m1"


def test_preflight_flag_off_vs_flag_on_reference_raises_missing_column(tmp_path):
    with pytest.raises(AssertionError, match=r"missing column\(s\) \['stoppage_source', 'stoppage_state'\]"):
        _preflight(tmp_path, stoppage=False)
