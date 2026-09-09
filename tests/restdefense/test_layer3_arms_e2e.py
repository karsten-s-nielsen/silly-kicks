"""TF-60 Task 11b: @e2e method gate for the Layer-3 arms (spec section 10).

Owner/fixture-gated: does NOT run in the normal ``-m "not e2e"`` suite. Asserts the METHOD (both arms
non-empty, RestDefenseGhostReport conservation reconciles, the merged table carries both arm columns,
and -- on SB360 -- a FOV companion drops below 1.0 on a cropped advanced-ball frame), NEVER metric
VALUES, so it stays compatible with the reported-not-gated convention.

The two fixtures below ``pytest.skip`` with an explicit reason when the owner's real-match data is not
wired (the ``test_tracking_real_data_sweep`` env-gated pattern; a silent skip hides breakage). Wire
``real_linked_tracking_match`` to a real full-coverage tracking match (native GK) and
``real_sb360_match`` to a real StatsBomb-360 match + roster at review time.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def fitted_corpus_xt():
    """A fitted ExpectedThreat for the e2e run. Owner may replace with a corpus-fit surface; the shared
    ramp xt is enough to exercise the arm METHOD (the e2e asserts method, not values)."""
    import numpy as np

    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


@pytest.fixture
def real_linked_tracking_match():
    """``(actions, frames, home_team_id)`` for a real full-coverage tracking match (native GK).

    Wire to the owner's loader at review time (e.g. ``scripts._loader_pining.load_matches`` behind the
    pining token). Skips cleanly until then."""
    if not os.environ.get("SK_LAYER3_E2E_TRACKING"):
        pytest.skip("SK_LAYER3_E2E_TRACKING not set; wire a real linked-tracking match to run this e2e.")
    # Owner wiring goes here (load actions/frames/home for the pointed match).
    pytest.skip("SK_LAYER3_E2E_TRACKING is set but the loader wiring is owner-provided at review time.")


@pytest.fixture
def real_sb360_match():
    """``(actions, frames, visible_area, roster, home_team_id)`` for a real StatsBomb-360 match."""
    if not os.environ.get("SK_LAYER3_E2E_SB360"):
        pytest.skip("SK_LAYER3_E2E_SB360 not set; wire a real SB360 match + roster to run this e2e.")
    pytest.skip("SK_LAYER3_E2E_SB360 is set but the loader wiring is owner-provided at review time.")


def _conserves(report) -> bool:
    return report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in


@pytest.mark.e2e
def test_layer3_arms_on_real_linked_tracking(real_linked_tracking_match, fitted_corpus_xt):
    from silly_kicks.restdefense import compute_rest_defense
    from silly_kicks.restdefense._arms import (
        merge_rest_defense,
        rest_defense_gk_deterrent,
        rest_defense_outfield_deterrent,
    )
    from silly_kicks.restdefense._columns import RD_GK_DETER_THREAT, RD_OUTFIELD_DETER_THREAT
    from silly_kicks.tracking import GhostGkModel, GhostOutfieldModel

    actions, frames, home = real_linked_tracking_match
    gk_arm, gk_rep = rest_defense_gk_deterrent(
        actions, frames, xt=fitted_corpus_xt, ghost_gk_model=GhostGkModel.from_variant("sweeper"), home_team_id=home
    )
    of_arm, of_rep = rest_defense_outfield_deterrent(
        actions,
        frames,
        xt=fitted_corpus_xt,
        ghost_outfield_model=GhostOutfieldModel.from_variant("default"),
        home_team_id=home,
    )
    assert len(gk_arm) > 0 and len(of_arm) > 0
    assert _conserves(gk_rep) and _conserves(of_rep)
    samples, _ = compute_rest_defense(actions, frames, xt=fitted_corpus_xt)
    merged = merge_rest_defense(samples, gk_arm, of_arm)
    assert {RD_GK_DETER_THREAT, RD_OUTFIELD_DETER_THREAT} <= set(merged.columns)


@pytest.mark.e2e
def test_layer3_arms_on_real_sb360(real_sb360_match, fitted_corpus_xt):
    from silly_kicks.keeper_identity import apply_keeper_identities_to_frames, resolve_keeper_identities
    from silly_kicks.restdefense import compute_rest_defense
    from silly_kicks.restdefense._arms import rest_defense_outfield_deterrent
    from silly_kicks.tracking import GhostOutfieldModel

    actions, frames, visible_area, roster, home = real_sb360_match
    keeper_map, _ = resolve_keeper_identities(actions, frames, identity="roster", roster=roster)
    frames = apply_keeper_identities_to_frames(frames, keeper_map)
    _of_arm, of_rep = rest_defense_outfield_deterrent(
        actions,
        frames,
        xt=fitted_corpus_xt,
        ghost_outfield_model=GhostOutfieldModel.from_variant("position_only"),
        home_team_id=home,
        visible_area=visible_area,
    )
    assert _conserves(of_rep)
    samples, _ = compute_rest_defense(actions, frames, xt=fitted_corpus_xt, visible_area=visible_area)
    # A genuinely cropped advanced-ball frame yields an observed fraction < 1.0 on a region companion.
    frac_cols = [c for c in samples.columns if c.endswith("_observed_fraction")]
    assert frac_cols, "no FOV observed-fraction companions -- visible_area was not threaded"
    assert any((samples[c] < 1.0).any() for c in frac_cols)
