"""Owner-gated coverage + provenance smoke for xT-GK's RAV completion model.

History (review H3 / ADR-024): accessible-space's xC is validated on OPEN-PLAY passes, but
xT-GK scores long aerial GOAL-KICKS -- a different completion regime. The owner-data OOD
smoke that used to live here measured ~31% goal-kick xC coverage and ESCALATED, which is
exactly what drove the goal-kick-coverage work: a scoped coordinate derivation
(``resolve_gk_geometry``) + a fitted ``GkCompletionModel`` that replaced get_xc. This file
now smokes the SHIPPED design on real goal-kicks: RAV resolves for ~all coord-resolvable
goal-kicks (the coverage win), every scored row carries machine-readable provenance, and the
origin-source mix is reported as a drift alarm (informational, NOT a pass/green assertion --
finiteness is not correctness; the sole green correctness gate is the native-origin pooled
out-of-fold calibration in scripts/train_gk_completion.py, recorded in the bundled model's
metrics.json).

This is NOT a CI gate: real full-match data with labelled goal-kick events is not committed
(the slim provider fixtures carry frames but not real goal-kick actions), so the test
self-skips unless the owner points XT_GK_E2E_MATCH_DIR at a directory holding:
  - frames.parquet      : long-form tracking frames (TRACKING_FRAMES_COLUMNS)
  - actions.parquet     : SPADL actions for the SAME match (must include goalkicks)
  - xt_corpus.parquet   : SPADL actions from a DISJOINT corpus to fit xT (no leakage)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

_MATCH_DIR = os.environ.get("XT_GK_E2E_MATCH_DIR")


@pytest.mark.skipif(
    not _MATCH_DIR,
    reason="set XT_GK_E2E_MATCH_DIR to a real match dir (frames/actions/xt_corpus parquet)",
)
def test_completion_coverage_and_provenance_on_real_goalkicks():
    import numpy as np
    import pandas as pd

    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking import XtGkReport, compute_xt_gk
    from silly_kicks.xthreat import ExpectedThreat

    d = Path(_MATCH_DIR)  # type: ignore[arg-type]
    frames = pd.read_parquet(d / "frames.parquet")
    actions = pd.read_parquet(d / "actions.parquet")
    corpus = pd.read_parquet(d / "xt_corpus.parquet")  # DISJOINT from `actions` (no leakage)

    xt = ExpectedThreat()
    xt.fit(corpus)

    # No [das] extra required: RAV now uses the bundled GS completion model.
    out = compute_xt_gk(actions, frames, xt=xt)
    is_gk = actions["type_id"].to_numpy() == spadlconfig.actiontype_id["goalkick"]
    gk = out.loc[is_gk]
    assert len(gk) > 0, "fixture must contain real goalkicks for the smoke to mean anything"

    # --- coverage win: RAV resolves for ~all goal-kicks with a resolvable destination ---
    # (rows whose destination cannot be resolved -- no native end, no in-period next-event --
    # are honestly NaN; everything else is now scored, vs the historical ~31% get_xc coverage.)
    dest_resolvable = gk["xt_gk_dest_source"].isin(["native", "next_event"]).to_numpy()
    if dest_resolvable.any():
        resolved = gk.loc[dest_resolvable, "xt_gk_rav"].notna().mean()
        assert resolved > 0.9, (
            f"goal-kick RAV unresolved for {1 - resolved:.0%} of dest-resolvable kicks "
            "-- the completion model should cover ~all of them"
        )
    assert gk["xt_gk_rav"].nunique() > 1, "goal-kick RAV is a single constant -- suspicious"

    # --- provenance: every scored row carries an origin + dest source tag ---
    scored = gk["xt_gk"].notna()
    assert gk.loc[scored, "xt_gk_origin_source"].notna().all()
    assert gk.loc[scored, "xt_gk_dest_source"].notna().all()

    # --- drift alarm (INFORMATIONAL, not a pass assertion) ---
    rep = XtGkReport.from_frame(out)
    composite_finite = float(np.isfinite(gk["xt_gk"].to_numpy()).mean())
    print(f"\n[xT-GK e2e] goalkicks={len(gk)}  composite_finite={composite_finite:.0%}")
    print(f"[xT-GK e2e] origin_source={rep.origin_source_counts}")
    print(f"[xT-GK e2e] dest_source={rep.dest_source_counts}")
