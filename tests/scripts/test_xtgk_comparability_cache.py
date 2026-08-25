"""Task 16 (ADR-068): _xtgk_comparability threads --cache-dir into load_matches so a match's
tracking artifact is downloaded once and reused, not fetched twice per run."""

import scripts._xtgk_comparability as xc


def test_collect_threads_cache_dir_to_loader(monkeypatch):
    seen: dict = {}

    def _fake_load_matches(**kwargs):
        seen.update(kwargs)
        return iter([])  # no matches -> _collect returns an empty frame; loader kwargs are the point

    monkeypatch.setattr(xc, "load_matches", _fake_load_matches)
    xc._collect(["skillcorner"], max_per_provider=6, tracking_limit=999999, xt=None, cache_dir="CACHE_SENTINEL")
    assert seen["cache_dir"] == "CACHE_SENTINEL"
    assert seen["providers"] == ["skillcorner"]
