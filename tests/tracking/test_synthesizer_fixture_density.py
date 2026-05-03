"""Fail-loud existence check for synthesized provider-input fixtures.

PR-S24 / lakehouse-review S3: closes the silent-vacuous-pass failure mode for
TF-12 per-period DOP-symmetry invariants. Runs BEFORE TF-12 invariants on
provider-bound CI matrix (alphabetical test order). Any provider missing >=1
shot AND >=1 keeper_save in EACH period of its synthesized output is a CI fail
with an actionable message.

Memory: feedback_invariant_testing (PR-S24 fail-loud-gate corollary).
Memory: feedback_synthesizer_shot_plus_keeper_save_pattern.
"""

from __future__ import annotations

import pytest

from silly_kicks.spadl import config as spadlconfig
from tests.tracking._provider_inputs import (
    load_provider_frames,
    synthesize_actions_per_period_dense,
)

_SHOT_TYPE_IDS = {spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick", "shot_penalty")}
_KEEPER_SAVE_TYPE_ID = spadlconfig.actiontype_id["keeper_save"]


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_per_period_shot_and_keeper_save_exist(provider):
    """For every period the synthesizer covers, assert >=1 shot AND >=1 keeper_save."""
    frames = load_provider_frames(provider)
    actions = synthesize_actions_per_period_dense(frames)
    periods = sorted(actions["period_id"].unique())
    assert len(periods) >= 1, f"{provider}: synthesizer produced zero periods of actions"
    for period in periods:
        in_period = actions[actions["period_id"] == period]
        n_shots = int(in_period["type_id"].isin(_SHOT_TYPE_IDS).sum())
        n_keeper_saves = int((in_period["type_id"] == _KEEPER_SAVE_TYPE_ID).sum())
        assert n_shots >= 1, (
            f"{provider} P{period}: synthesized actions have NO shot rows. "
            "Required for TF-12 invariants per feedback_synthesizer_shot_plus_keeper_save_pattern."
        )
        assert n_keeper_saves >= 1, (
            f"{provider} P{period}: synthesized actions have NO keeper_save rows. "
            "Required for TF-12 invariants per feedback_synthesizer_shot_plus_keeper_save_pattern."
        )


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner", "pff"])
def test_synthesized_action_id_globally_unique(provider):
    """Renumbering across periods must yield unique action_id."""
    frames = load_provider_frames(provider)
    actions = synthesize_actions_per_period_dense(frames)
    assert actions["action_id"].is_unique, f"{provider}: action_id not globally unique after concat"
