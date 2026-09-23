"""The xCross causal validator's loader seam (spec section 4.5): refs + load_match, resumable.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path.
"""

from __future__ import annotations

import _loader_pining as lp
import pandas as pd
from _fake_corpus import SpyLoader, install_fake_corpus, make_loaded, make_ref

import scripts.validate_xcross_causal as mod


def test_run_walks_refs_via_load_and_resumes(tmp_path, monkeypatch):
    import silly_kicks.causal.opportunities as opp

    # A cheap opportunity stub whose carrier coverage is below the gate -> the run takes the
    # `no_eligible_provider` branch and never reaches the (heavy) causal analysis.
    monkeypatch.setattr(
        opp, "build_opportunities", lambda frames, actions, **k: pd.DataFrame({"carrier_resolved": [False]})
    )
    refs = [make_ref("gradientsports", "m0"), make_ref("gradientsports", "m1")]
    spy = SpyLoader({r.key: make_loaded(*r.key) for r in refs})
    install_fake_corpus(monkeypatch, lp, refs=refs, loader=spy)

    m = mod.run(tmp_path, ["gradientsports"], carrier_min=0.6, seed=0)
    assert m["status"] == "no_eligible_provider"
    assert len(spy.calls) == 2  # one load per match

    mod.run(tmp_path, ["gradientsports"], carrier_min=0.6, seed=0)
    assert len(spy.calls) == 2, "resume re-loaded a match whose shard already existed"
