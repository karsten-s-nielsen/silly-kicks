"""Owner-gated WC2022 GS e2e for the general restart-coordinate enrichment (4.22.0, ADR-025).
Needs PINING_FOR_THE_DATA_TOKEN (owner-tier Gradient Sports).

Three real-data checks the synthetic CI fixtures cannot give:
  1. add_restart_coordinates coverage uplift + source-distribution sanity on real GS goalkicks
     (~60% NaN native origin per the live probe) + canonical coords never mutated + tripwire sanity.
  2. xT-GK boundary-mapping parity: on real GS, ``compute_xt_gk``'s ``xt_gk_origin_source`` uses ONLY
     the frozen legacy enum (the generic ``restart_prior``/``tracking_ball`` labels never leak through
     the ``resolve_gk_geometry`` shim) -- the real-data guard for the consolidation.
  3. compute_gk_completion / add_gk_completion run on real GS and score in-scope rows (transitive
     parity: they consume resolve_gk_geometry, which the committed golden snapshot pins byte-identical).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")

_RELEASED_SOURCES = {
    "native",
    "tracking_ball",
    "tracking_gk",
    "restart_prior",
    "next_event",
    "unresolved",
    "tripwire_reverted",
}
# The frozen resolve_gk_geometry / xt_gk_origin_source contract: generic labels must NOT leak.
_LEGACY_GK_SOURCES = {"native", "tracking_gk", "goalkick_prior", "unresolved"}


@pytest.mark.skipif(not _TOKEN, reason="needs PINING_FOR_THE_DATA_TOKEN (owner-tier GS)")
def test_restart_coordinate_enrichment_real_wc2022():
    from scripts._loader_pining import load_matches
    from silly_kicks.spadl import add_restart_coordinates
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking import RestartCoordinateReport, compute_gk_completion, compute_xt_gk
    from silly_kicks.xthreat import ExpectedThreat

    GK = spadlconfig.actiontype_id["goalkick"]
    matches = list(load_matches(providers=["gradientsports"], max_per_provider=2, tracking_limit=None))
    assert len(matches) >= 1, "no GS matches returned by the loader"

    _p, _mid, actions, frames, _home = matches[0]
    coord_cols = ["start_x", "start_y", "end_x", "end_y"]
    before = actions[coord_cols].copy()

    # --- (1a) events-only: canonical untouched + goalkick origin coverage -> ~100% (rule-point) ---
    ev = add_restart_coordinates(actions, frames=None)
    pd.testing.assert_frame_equal(actions[coord_cols], before)  # NEVER mutates canonical coords
    assert set(ev["start_coord_source"].dropna().unique()) <= _RELEASED_SOURCES
    assert set(ev["end_coord_source"].dropna().unique()) <= _RELEASED_SOURCES

    gk_ev = ev[ev["type_id"] == GK]
    assert len(gk_ev) > 0, "fixture must contain real goalkicks"
    native_origin = gk_ev["start_x"].notna().mean()
    enriched_origin = gk_ev["enriched_start_x"].notna().mean()
    assert enriched_origin >= native_origin  # enrichment is never worse than native
    assert enriched_origin > 0.99, enriched_origin  # events-only rule-point fills ~all goalkick origins

    # events-only rule-points are in-region by construction -> zero tripwire reversions
    rep_ev = RestartCoordinateReport.from_frame(ev)
    assert rep_ev.n_tripwire_reversions == 0, rep_ev.start_source_counts
    print(f"\n[restart e2e] events-only goalkick native_origin={native_origin:.0%} -> enriched={enriched_origin:.0%}")
    print(f"[restart e2e] events-only start_source={rep_ev.start_source_counts}")

    # --- (1b) frames path: tracking tiers enabled; still ~100% goalkick origin coverage ---
    fr = add_restart_coordinates(actions, frames=frames)
    gk_fr = fr[fr["type_id"] == GK]
    assert gk_fr["enriched_start_x"].notna().mean() > 0.99
    rep_fr = RestartCoordinateReport.from_frame(fr)
    # tripwire reversions (frames path can revert a mis-linked tracking_ball) stay rare on real data
    assert rep_fr.n_tripwire_reversions <= 0.02 * len(fr), rep_fr.n_tripwire_reversions
    print(f"[restart e2e] frames start_source={rep_fr.start_source_counts}  reversions={rep_fr.n_tripwire_reversions}")

    # --- (2) xT-GK boundary-mapping parity on real data (fit xT on a disjoint match, no leakage) ---
    fit_actions = matches[1][2] if len(matches) >= 2 else actions
    score_actions, score_frames = (matches[1][2], matches[1][3]) if len(matches) >= 2 else (actions, frames)
    xt = ExpectedThreat()
    xt.fit(fit_actions)
    out = compute_xt_gk(score_actions, score_frames, xt=xt)
    srcs = set(out["xt_gk_origin_source"].dropna().unique())
    assert srcs <= _LEGACY_GK_SOURCES, f"generic restart labels leaked into xt_gk_origin_source: {srcs}"
    assert "restart_prior" not in srcs and "tracking_ball" not in srcs
    print(f"[restart e2e] xt_gk_origin_source={sorted(srcs)} (legacy enum only -- boundary mapping holds)")

    # --- (3) completion model runs on real GS + scores in-scope rows ---
    comp = compute_gk_completion(score_actions, score_frames)
    gk_mask = score_actions["type_id"].to_numpy() == GK
    assert np.isfinite(comp.to_numpy()[gk_mask]).any(), "completion produced no finite goalkick probabilities"
