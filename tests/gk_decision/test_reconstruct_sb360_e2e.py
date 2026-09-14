"""End-to-end SB360 reconstruction (TF-62 Phase 2): a GK goal-kick through the REAL producer +
the actor-identity bridge + the bundled PassCompletionModel + the engine.

Anonymous (row-numbered) freeze-frame ids + an ``is_actor`` keeper row -> the bridge stamps the real
keeper id, so ReconstructedOptionSet can key on it. Committed synthetic fixture (small; per-action-LTR:
the acting keeper's team attacks x=105), so this runs in the regular suite -- not an e2e dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet, compute_gk_decision_value
from silly_kicks.keeper_identity import apply_actor_identities_to_frames
from silly_kicks.tracking import gk_distribution_mask, snapshot_to_tracking_frames

_KEEPER_ID = 777  # the real acting keeper id carried on the SPADL action (frames are anonymous)
_GOALKICK = 22  # SPADL type_id


def _actions():
    return pd.DataFrame(
        [
            dict(
                game_id="m1",
                period_id=1,
                action_id=1,
                time_seconds=10.0,
                type_id=_GOALKICK,
                player_id=_KEEPER_ID,
                team_id=1,
                start_x=8.0,
                start_y=34.0,
                end_x=30.0,
                end_y=34.0,
            )
        ]
    )


def _snapshot():
    # per-action-LTR: keeper's team (1) attacks x=105. Anonymous player ids (0..) as the real port emits.
    # keeper (actor) deep; 3 reachable home teammates; 1 opponent -> exactly one opponent team.
    return pd.DataFrame(
        [
            dict(action_id=1, team_id=1, player_id=0, is_goalkeeper=True, is_actor=True, x=8.0, y=34.0),
            dict(action_id=1, team_id=1, player_id=1, is_goalkeeper=False, is_actor=False, x=30.0, y=34.0),
            dict(action_id=1, team_id=1, player_id=2, is_goalkeeper=False, is_actor=False, x=45.0, y=20.0),
            dict(action_id=1, team_id=1, player_id=3, is_goalkeeper=False, is_actor=False, x=25.0, y=50.0),
            dict(action_id=1, team_id=2, player_id=4, is_goalkeeper=False, is_actor=False, x=20.0, y=34.0),
        ]
    )


def _visible_area():
    # a full-pitch polygon -> the keeper neighbourhood is fully observed (FOV gate passes)
    return pd.DataFrame(
        {"action_id": [1], "polygon": [np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])]}
    )


def test_sb360_goalkick_scores_through_producer_bridge_and_bundled_xpass():
    from silly_kicks.expected_passing import PassCompletionModel

    actions = _actions()
    frames, _links = snapshot_to_tracking_frames(_snapshot(), actions)
    # SB360 freeze-frames are anonymous -> stamp the acting keeper's real id onto the actor row (ADR-078).
    frames = apply_actor_identities_to_frames(frames, actions)

    gk_actions = actions[gk_distribution_mask(actions, frames, resolve_gk="robust").to_numpy()]
    assert len(gk_actions) == 1  # the goal-kick is in the GK-distribution domain

    os_ = ReconstructedOptionSet(
        gk_actions,
        frames,
        xpass=PassCompletionModel.bundled(),
        params=GkDecisionParams(reachability_min_xpass=0.0),  # keep every reachable option in this small scene
        keeper_ids=[_KEEPER_ID],
        frame_convention="per_action_ltr",
        visible_area=_visible_area(),
    )
    samples, report = compute_gk_decision_value(os_)

    assert report.n_decisions_in == 1
    assert report.n_scored == 1  # 3 reachable teammates + chosen >= min_options
    assert (
        report.n_scored
        + report.n_too_few_options
        + report.n_no_unique_chosen
        + report.n_chosen_unvalued
        + report.n_no_frame
        + report.n_fov_cropped
        == report.n_decisions_in
    )
    row = samples.iloc[0]
    assert row["option_set_source"] == "reconstructed"
    assert row["keeper"] == str(_KEEPER_ID)  # canonical real id (the bridge worked), not an anonymous row number
    assert 0.0 <= row["sel_efficiency"] <= 1.0
    assert 0.0 <= row["decision_pct"] <= 1.0
    assert np.isfinite(row["chosen_ev"]) and np.isfinite(row["decision_value"])
