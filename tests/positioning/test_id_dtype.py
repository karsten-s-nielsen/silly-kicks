"""TF-56 id-dtype invariance (ADR-019): compute_positioning_gap on string ids scores coherently."""

from __future__ import annotations

import pandas as pd


def test_string_team_and_player_ids_score_coherently(one_frame, fitted_xt):
    from silly_kicks.positioning import compute_positioning_gap

    f = one_frame.copy()
    f["team_id"] = f["team_id"].map(lambda t: str(int(t)) if pd.notna(t) else t)
    f["player_id"] = f["player_id"].map(lambda p: str(int(p)) if pd.notna(p) else p)

    samples, report = compute_positioning_gap(f, xt=fitted_xt)
    assert report.conserves()
    scored = samples[samples["positioning_gap_source"] == "scored"]
    assert len(scored) == 1  # object-dtype ids resolve identically (id_compat, ADR-019)
    assert float(scored["positioning_gap"].iloc[0]) >= 0.0
    # the emitted defending team_id is the RAW (string) id, not a canonicalised surrogate
    assert scored["team_id"].iloc[0] == "1"


def test_numeric_and_string_ids_give_the_same_scored_gap(one_frame, fitted_xt):
    from silly_kicks.positioning import compute_positioning_gap

    num_samples, _ = compute_positioning_gap(one_frame, xt=fitted_xt)
    f = one_frame.copy()
    f["team_id"] = f["team_id"].map(lambda t: str(int(t)) if pd.notna(t) else t)
    f["player_id"] = f["player_id"].map(lambda p: str(int(p)) if pd.notna(p) else p)
    str_samples, _ = compute_positioning_gap(f, xt=fitted_xt)

    num_gap = float(num_samples.loc[num_samples["positioning_gap_source"] == "scored", "positioning_gap"].iloc[0])
    str_gap = float(str_samples.loc[str_samples["positioning_gap_source"] == "scored", "positioning_gap"].iloc[0])
    assert num_gap == str_gap  # the gap is invariant to id dtype
