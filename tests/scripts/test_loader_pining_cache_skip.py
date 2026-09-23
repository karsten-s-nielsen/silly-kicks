"""_loader_pining_to_cache resumes on the load seam (ADR-052 D14 / ADR-068).

It lists wanted refs up front, skips any already MATERIALIZED (`_cached` pre-filter, a metadata-only
loop over refs), and loads ONLY the missing ones through a resume-before-load ``for_each`` pass. A
frame set with no velocity is a counted EXCLUSION, never a fabricated cache entry.
"""

from __future__ import annotations

import sys

import _loader_pining as lp
import _loader_pining_to_cache as cache_mod
import pandas as pd
from _fake_corpus import SpyLoader, install_fake_corpus, make_loaded, make_ref


def _write_complete_cache(out, provider, match_id):
    """Both unconditional artifacts, in write_match_cache's order (frames THEN meta) -> COMPLETE."""
    gdir = out / provider / str(match_id)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "frames.parquet").write_bytes(b"x")
    (gdir / "meta.json").write_text("{}")


def _frames_with_velocity(match_id):
    return pd.DataFrame({"vx": [0.0], "vy": [0.0], "game_id": [str(match_id)], "is_goalkeeper": [False]})


def test_resume_skips_cached_and_loads_only_missing(monkeypatch, tmp_path):
    _write_complete_cache(tmp_path, "skillcorner", "1")  # already materialized -> not re-fetched
    refs = [make_ref("skillcorner", "1"), make_ref("skillcorner", "2"), make_ref("gradientsports", "9")]
    loader = SpyLoader(
        {
            ("skillcorner", "2"): make_loaded("skillcorner", "2", frames=_frames_with_velocity("2")),
            ("gradientsports", "9"): make_loaded("gradientsports", "9", frames=_frames_with_velocity("9")),
        }
    )
    install_fake_corpus(monkeypatch, lp, refs=refs, loader=loader)
    monkeypatch.setattr(sys, "argv", ["prog", "--providers", "skillcorner", "gradientsports", "--out", str(tmp_path)])

    cache_mod.main()

    loaded = {k for k, _ in loader.calls}
    assert ("skillcorner", "1") not in loaded  # cached -> never loaded
    assert loaded == {("skillcorner", "2"), ("gradientsports", "9")}
    assert (tmp_path / "skillcorner" / "2" / "frames.parquet").exists()
    assert (tmp_path / "gradientsports" / "9" / "meta.json").exists()


def test_all_cached_short_circuits_without_loading(monkeypatch, tmp_path):
    _write_complete_cache(tmp_path, "skillcorner", "1")
    loader = SpyLoader({})
    install_fake_corpus(monkeypatch, lp, refs=[make_ref("skillcorner", "1")], loader=loader)
    monkeypatch.setattr(sys, "argv", ["prog", "--providers", "skillcorner", "--out", str(tmp_path)])

    cache_mod.main()
    assert loader.calls == []  # everything cached -> nothing loaded (no re-fetch)


def test_partial_cache_from_crash_between_writes_is_redone(monkeypatch, tmp_path):
    # A crash between write_match_cache's two writes leaves frames.parquet WITHOUT meta.json. _cached
    # must NOT treat that as complete, else home_team_id (in the missing meta.json) is lost forever.
    (tmp_path / "skillcorner" / "1").mkdir(parents=True)
    (tmp_path / "skillcorner" / "1" / "frames.parquet").write_bytes(b"x")  # meta.json NOT written
    assert cache_mod._cached(tmp_path, "skillcorner", "1") is False

    loader = SpyLoader({("skillcorner", "1"): make_loaded("skillcorner", "1", frames=_frames_with_velocity("1"))})
    install_fake_corpus(monkeypatch, lp, refs=[make_ref("skillcorner", "1")], loader=loader)
    monkeypatch.setattr(sys, "argv", ["prog", "--providers", "skillcorner", "--out", str(tmp_path)])

    cache_mod.main()
    assert ("skillcorner", "1") in {k for k, _ in loader.calls}  # re-fetched, not silently skipped
    assert (tmp_path / "skillcorner" / "1" / "meta.json").exists()  # now COMPLETE


def test_no_velocity_match_is_excluded_not_cached(monkeypatch, tmp_path):
    # A frame set with no vx/vy cannot serve the trainer: it is a counted EXCLUSION, never materialized.
    loader = SpyLoader({("skillcorner", "1"): make_loaded("skillcorner", "1", frames=pd.DataFrame({"game_id": ["1"]}))})
    install_fake_corpus(monkeypatch, lp, refs=[make_ref("skillcorner", "1")], loader=loader)
    monkeypatch.setattr(sys, "argv", ["prog", "--providers", "skillcorner", "--out", str(tmp_path)])

    cache_mod.main()
    assert ("skillcorner", "1") in {k for k, _ in loader.calls}  # attempted
    assert not (tmp_path / "skillcorner" / "1" / "frames.parquet").exists()  # but not materialized
