"""The sigma-sweep tuner's loader seam (spec section 4.5): refs + load_match, resumable.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import pandas as pd
from _fake_corpus import SpyLoader, install_fake_corpus, make_loaded, make_ref

import scripts.tune_structural_pass_sigma as mod


def test_main_walks_refs_via_load_and_resumes(tmp_path, monkeypatch):
    refs = [make_ref("gradientsports", "m0"), make_ref("gradientsports", "m1")]
    spy = SpyLoader({r.key: make_loaded(*r.key) for r in refs})
    install_fake_corpus(monkeypatch, mod, refs=refs, loader=spy)
    # Stub the per-match sweep + the report so the test exercises the SEAM, not the sigma kernel.
    monkeypatch.setattr(mod, "_match_records", lambda item: pd.DataFrame({"sigma": [15.0], "sgm": [1.0]}))
    monkeypatch.setattr(mod, "report", lambda df: None)

    mod.main(2, str(tmp_path))
    assert len(spy.calls) == 2  # one load per match
    mod.main(2, str(tmp_path))
    assert len(spy.calls) == 2, "resume re-loaded a match whose shard already existed"
