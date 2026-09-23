"""The two trainers' loader seam (spec section 4.5): `_pining_source` / `_source_key` / `_extract(load=)`.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path, so bare imports resolve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import train_xcross_attempt as xc
import train_xshot_occurrence as xs
from _fake_corpus import make_loaded, make_ref


@pytest.mark.parametrize("mod", [xs, xc])
def test_source_key_handles_both_a_ref_and_a_data_dir_tuple(mod):
    """A MatchRef (pining) keys by its `.key`; a --data-dir tuple keys by (provider, match_id)."""
    assert mod._source_key(make_ref("gradientsports", "10502")) == ("gradientsports", "10502")
    # a --data-dir item is a plain (provider, match_id, actions, frames, home) tuple, no `.key`
    assert mod._source_key(("gradientsports", 10502, None, None, None)) == ("gradientsports", "10502")


def _fake_xshot_prepare(monkeypatch):
    import silly_kicks.tracking._xshot_occurrence as xo

    def _prep(frames, actions, **kw):
        x = pd.DataFrame({name: [0.0] for name in xo.XSHOT_FEATURE_NAMES_FAITHFUL})
        return x, np.array([1]), np.array(["g0"])

    monkeypatch.setattr(xo, "prepare_xshot_training_data", _prep)


def test_xshot_extract_walks_refs_via_load_and_resumes(tmp_path, monkeypatch):
    _fake_xshot_prepare(monkeypatch)
    refs = [make_ref("gradientsports", "m0"), make_ref("gradientsports", "m1")]
    calls: list = []

    def _load(ref):
        calls.append(ref.key)
        return make_loaded(ref.provider, ref.match_id, actions=pd.DataFrame(), frames=pd.DataFrame(), home_team_id=1)

    shard_root = tmp_path / "shards"
    X, *_ = xs._extract(refs, 5.0, shard_root=shard_root, load=_load)
    assert len(X) == 2 and len(calls) == 2  # one loaded row per match

    n_first = len(calls)
    xs._extract(refs, 5.0, shard_root=shard_root, load=_load)
    assert len(calls) == n_first, "resume re-loaded a match whose shard already existed"
