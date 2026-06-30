"""compute_xt_gk provider-distrust wiring + coherence + S4 report + C1 (CR 2026-06-30)."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking._gk_completion import GkCompletionModel
from silly_kicks.tracking._xt_gk import XtGkReport, compute_xt_gk
from tests.tracking.test_gk_completion_pertype_gate import _fitted_model_with_gate as _gate_model

from ._xt_gk_fixtures import make_fitted_xt, make_skillcorner_case


# --------------------------------------------------------------------------------------
# Task 5: provider-distrust wiring
# --------------------------------------------------------------------------------------
def test_skillcorner_goalkick_origin_is_tracked_not_native():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    out = compute_xt_gk(actions, frames, xt=make_fitted_xt())
    gk = out[actions["type_id"].to_numpy() == 22].iloc[0]
    assert gk["xt_gk_origin_source"] == "tracking_gk"
    assert abs(gk["xt_gk_origin_x"] - 5.5) < 1e-6
    assert gk["xt_gk_origin_x"] != 25.0


def test_native_goalkick_out_of_region_column_emitted():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    out = compute_xt_gk(actions, frames, xt=make_fitted_xt())
    assert "xt_gk_native_goalkick_out_of_region" in out.columns


# --------------------------------------------------------------------------------------
# Task 6: coherence -- resolved origin feeds base + pressure + RAV/completion
# --------------------------------------------------------------------------------------
def test_resolved_origin_feeds_pressure_and_rav_not_only_base():
    # distrusted GOAL-KICK whose tracked-keeper origin moves (5.5 vs 12.0, both in-box) -> every
    # origin-derived term must respond. A model-served completion (gated to "model") keeps the RAV
    # path origin-dependent (SkillCorner goal-kicks otherwise serve base_rate -> origin-independent).
    model = _gate_model({"goalkick": "model", "throw_in": "model", "other": "model"})
    a1, f1 = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5, defender_near=(5.5, 34.0))
    a2, f2 = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=12.0, defender_near=(5.5, 34.0))
    o1 = compute_xt_gk(a1, f1, xt=make_fitted_xt(), completion=model)
    o2 = compute_xt_gk(a2, f2, xt=make_fitted_xt(), completion=model)
    r1 = o1[o1["xt_gk"].notna()].iloc[0]
    r2 = o2[o2["xt_gk"].notna()].iloc[0]
    assert r1["xt_gk_origin_x"] == 5.5 and r2["xt_gk_origin_x"] == 12.0  # resolved origin moved
    assert r1["xt_gk_pressure"] != r2["xt_gk_pressure"]  # origin -> pressure
    assert r1["xt_gk_base"] != r2["xt_gk_base"]  # origin -> base
    assert r1["xt_gk_rav"] != r2["xt_gk_rav"]  # origin -> RAV/completion


# --------------------------------------------------------------------------------------
# Task 7: XtGkReport count
# --------------------------------------------------------------------------------------
def test_report_counts_out_of_region_flags():
    df = pd.DataFrame(
        {
            "xt_gk_origin_source": ["native", "tracking_gk"],
            "xt_gk_dest_source": ["native", "next_event"],
            "xt_gk": [0.1, 0.2],
            "xt_gk_native_goalkick_out_of_region": [True, False],
        }
    )
    rep = XtGkReport.from_frame(df)
    assert rep.n_native_goalkick_out_of_region == 1


# --------------------------------------------------------------------------------------
# C1: one-call-one-match enforced uniformly (escape hatch removed)
# --------------------------------------------------------------------------------------
def test_multi_provider_raises_even_with_completion_override():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    frames = frames.copy()
    frames.loc[frames.index[: len(frames) // 2], "source_provider"] = "gradientsports"  # 2 real providers
    model = GkCompletionModel.from_variant("default")
    with pytest.raises(ValueError, match="multiple real providers"):
        compute_xt_gk(actions, frames, xt=make_fitted_xt(), completion=model)


def test_single_provider_with_completion_override_still_works():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    model = GkCompletionModel.from_variant("default")
    out = compute_xt_gk(actions, frames, xt=make_fitted_xt(), completion=model)  # no raise
    assert "xt_gk" in out.columns
