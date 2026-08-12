"""Owner-gated WC2022 GS e2e for structural_pass (TF-45). Needs PINING_FOR_THE_DATA_TOKEN.

The LBS-AUC assertion is a correctness/regression guard, NOT a reproduction of the
paper's progression finding (structural_lbs > 0 <=> forward pass, so it is partly
tautological). All validation metrics use open-play successful `pass` rows only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")


@pytest.mark.skipif(not _TOKEN, reason="needs PINING_FOR_THE_DATA_TOKEN (owner-tier GS)")
def test_structural_pass_real_wc2022():
    from scripts._loader_pining import load_matches
    from silly_kicks.tracking.features import add_structural_pass

    parts = []
    for _p, _mid, actions, frames, _home in load_matches(
        providers=["gradientsports"], max_per_provider=2, tracking_limit=None
    ):
        # Enrich on the FULL action stream, THEN filter -- pre-filtering before the
        # link hides dropped actions from the ADR-017 coverage guard, producing
        # spurious low-coverage warnings.
        enriched = add_structural_pass(actions, frames)  # type: ignore[arg-type]
        passes = enriched[(enriched["type_id"] == 0) & (enriched["result_id"] == 1)].copy()
        passes["enters_third"] = (passes["start_x"] < 70.0) & (passes["end_x"] >= 70.0)
        parts.append(passes)
    df = pd.concat(parts, ignore_index=True)
    valid = df[df["structural_lbs"].notna()].copy()
    assert len(valid) > 500

    # 1. base-rate band (paper-consistent territorial-progression frequency)
    base = valid["enters_third"].mean()
    assert 0.07 <= base <= 0.13, base

    # 2. LBS regression guard (tautological, NOT a paper reproduction)
    from sklearn.metrics import roc_auc_score

    lab = valid["enters_third"].to_numpy(bool)
    auc = roc_auc_score(lab, valid["structural_lbs"].to_numpy(float))
    assert auc >= 0.70, auc

    # 3. targeted coordinate-frame invariant -- predicate-selected at runtime (no frozen id)
    fwd = valid[(valid["end_x"] - valid["start_x"] > 25.0) & (valid["structural_lbs"] >= 1)]
    assert len(fwd) > 0, "expected forward passes with >=1 bypassed defender"

    # 4. SGM conditioning at sigma=15 (concrete ceilings; a drift to sigma=12 would trip)
    sgm = valid["structural_sgm"].to_numpy(float)
    sgm = sgm[np.isfinite(sgm)]
    assert np.abs(sgm).max() <= 200.0, np.abs(sgm).max()
    assert np.percentile(np.abs(sgm), 99) <= 20.0
