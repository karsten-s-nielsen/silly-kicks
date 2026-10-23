"""The two trainers' loader seam (spec section 4.5): `_pining_source` / `_source_key` / `_extract(load=)`.

ADR-129 moved `_source_key` (with the extract stage) into `scripts/_xshot_stages.py` / `_xcross_stages.py`.

tests/scripts/ has NO __init__.py; conftest puts scripts/ on sys.path, so bare imports resolve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import train_xshot_occurrence as xs
from _fake_corpus import make_ref

from scripts import _xcross_stages as xcs
from scripts import _xshot_stages as xss


@pytest.mark.parametrize("mod", [xss, xcs])
def test_source_key_handles_both_a_ref_and_a_data_dir_tuple(mod):
    """A MatchRef (pining) keys by its `.key`; a --data-dir tuple keys by (provider, match_id)."""
    assert mod._source_key(make_ref("gradientsports", "10502")) == ("gradientsports", "10502")
    # a --data-dir item is a plain (provider, match_id, actions, frames, home) tuple, no `.key`
    assert mod._source_key(("gradientsports", 10502, None, None, None)) == ("gradientsports", "10502")


def test_xshot_extract_walks_refs_via_load_and_resumes(tmp_path, monkeypatch):
    """ADR-128: the extraction is the `extract` ItemStage (refs -> load -> work through `for_each`); a
    re-run never re-extracts a match whose shard already exists."""
    import silly_kicks.tracking._xshot_occurrence as xo
    from scripts._stages import ItemStage, LocalStore, make_context, reduce_stage, run_unit, sorted_units
    from scripts._toy_corpora import build_xshot_corpus, xshot_arms
    from tests.scripts._stage_harness import PROV, params_from_args

    calls: list = []

    def _prep(frames, shots, **kw):
        calls.append(frames["game_id"].iloc[0])
        x = pd.DataFrame({name: [0.0] for name in xo.XSHOT_FEATURE_NAMES_FAITHFUL})
        return x, np.array([1]), np.array(["g0"])

    monkeypatch.setattr(xo, "prepare_xshot_training_data", _prep)
    build_xshot_corpus(tmp_path / "d" / "toyxs", providers=("gradientsports",) * 2, episodes=1, heavy_episodes=None)
    args = ["--dataset", "toyxs", "--arms-json", xshot_arms(tmp_path / "d" / "toyxs", paired=False)]
    ctx = make_context(
        xs.PIPELINE,
        params_from_args(xs.PIPELINE, args),
        PROV,
        store=LocalStore(tmp_path / "s"),
        data_root=tmp_path / "d",
    )
    st = xs.PIPELINE.stage("extract")
    assert isinstance(st, ItemStage)
    refs = st.ref_map(ctx)
    for _ in range(2):
        for key in sorted_units(ctx, st):
            run_unit(ctx, st, key, worker="test", ref=refs[key])
    assert len(calls) == 2, f"one extraction per match, and none on the re-run: {calls}"
    reduce_stage(ctx, "extract")
    assert len(ctx.upstream("extract")) == 2  # one row per match
