"""Task 19 (ADR-052/ADR-068): _loader_pining_to_cache resumes -- it lists wanted matches up front,
skips any already cached, and re-fetches ONLY the missing ones (never re-downloading the corpus)."""

import sys

import _loader_pining as lp
import _loader_pining_to_cache as cache_mod


def _write_complete_cache(out, provider, match_id):
    """Both unconditional artifacts, in write_match_cache's order (frames THEN meta) -> a COMPLETE cache."""
    gdir = out / provider / str(match_id)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "frames.parquet").write_bytes(b"x")
    (gdir / "meta.json").write_text("{}")


def test_resume_skips_cached_and_drops_empty_providers(monkeypatch, tmp_path):
    # Pre-cache skillcorner/1 COMPLETELY (_cached requires BOTH frames.parquet AND meta.json).
    _write_complete_cache(tmp_path, "skillcorner", "1")

    monkeypatch.setattr(
        lp,
        "select_match_ids",
        lambda **kw: [("skillcorner", "1"), ("skillcorner", "2"), ("gradientsports", "9")],
    )
    seen: dict = {}

    def _fake_load_matches(**kwargs):
        seen.update(kwargs)
        return iter([])  # nothing to write; the point is WHICH ids get fetched

    monkeypatch.setattr(lp, "load_matches", _fake_load_matches)
    monkeypatch.setattr(sys, "argv", ["prog", "--providers", "skillcorner", "gradientsports", "--out", str(tmp_path)])

    cache_mod.main()

    # The already-cached skillcorner/1 is NOT re-fetched; only the missing ids are passed.
    assert seen["match_ids"] == {"skillcorner": ["2"], "gradientsports": ["9"]}
    # Both providers still have work; neither is passed with an empty list (the manifest-expansion trap).
    assert set(seen["providers"]) == {"skillcorner", "gradientsports"}
    assert seen["max_per_provider"] is None  # already applied by select_match_ids


def test_all_cached_short_circuits_without_loading(monkeypatch, tmp_path):
    _write_complete_cache(tmp_path, "skillcorner", "1")
    monkeypatch.setattr(lp, "select_match_ids", lambda **kw: [("skillcorner", "1")])

    called = {"n": 0}

    def _fake_load_matches(**kwargs):
        called["n"] += 1
        return iter([])

    monkeypatch.setattr(lp, "load_matches", _fake_load_matches)
    monkeypatch.setattr(sys, "argv", ["prog", "--providers", "skillcorner", "--out", str(tmp_path)])

    cache_mod.main()
    assert called["n"] == 0  # everything cached -> load_matches never called (no re-fetch)


def test_partial_cache_from_crash_between_writes_is_redone(monkeypatch, tmp_path):
    # A crash/OOM between write_match_cache's two writes leaves frames.parquet WITHOUT meta.json.
    # _cached must NOT treat that as complete -- else the match is skipped forever and home_team_id
    # (in the missing meta.json) is lost, surfacing only much later in the trainer.
    (tmp_path / "skillcorner" / "1").mkdir(parents=True)
    (tmp_path / "skillcorner" / "1" / "frames.parquet").write_bytes(b"x")  # meta.json NOT written
    assert cache_mod._cached(tmp_path, "skillcorner", "1") is False

    monkeypatch.setattr(lp, "select_match_ids", lambda **kw: [("skillcorner", "1")])
    seen: dict = {}

    def _fake_load_matches(**kwargs):
        seen.update(kwargs)
        return iter([])

    monkeypatch.setattr(lp, "load_matches", _fake_load_matches)
    monkeypatch.setattr(sys, "argv", ["prog", "--providers", "skillcorner", "--out", str(tmp_path)])

    cache_mod.main()
    # The partially-written match IS re-fetched (not silently skipped).
    assert seen["match_ids"] == {"skillcorner": ["1"]}
