"""Conservation over the reconstruction path (TF-62 Phase 2): the adapter's no_frame / fov_cropped
drops are threaded into GkDecisionReport via extra_drops so n_decisions_in stays the TRUE population."""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet, compute_gk_decision_value


class _FakeXPass:
    def predict_completion(self, ox, oy, tx, ty):
        ox, oy, tx, ty = (np.asarray(a, dtype=float) for a in (ox, oy, tx, ty))
        return np.clip(1.0 - np.hypot(tx - ox, ty - oy) / 120.0, 0.0, 1.0)


def _keeper_row(frame_id, x=10.0):
    return dict(
        game_id="g",
        period_id=1,
        frame_id=frame_id,
        player_id=99,
        team_id=7,
        x=x,
        y=34.0,
        is_ball=False,
        is_goalkeeper=True,
        is_actor=True,
    )


def _p(fid, pid, team, x, y, gk=False):
    return dict(
        game_id="g",
        period_id=1,
        frame_id=fid,
        player_id=pid,
        team_id=team,
        x=x,
        y=y,
        is_ball=False,
        is_goalkeeper=gk,
        is_actor=False,
    )


def _ball(fid):
    return dict(
        game_id="g",
        period_id=1,
        frame_id=fid,
        player_id=1,
        team_id=None,
        x=10.0,
        y=34.0,
        is_ball=True,
        is_goalkeeper=False,
        is_actor=False,
    )


def test_reconstruction_report_conserves_across_all_drop_reasons():
    # Four GK decisions, one per drop class + one scored (per_action_ltr: frame_id == action_id):
    #   d1 (action_id 1) SCORED  -- keeper + 3 reachable teammates + opponent; polygon observes keeper.
    #   d2 (action_id 2) FOV_CROPPED -- keeper neighbourhood not observed by its visible_area polygon.
    #   d3 (action_id 3) TOO_FEW_OPTIONS -- keeper + 1 teammate + opponent (chosen + 1 alt = 2 < 3).
    #   d4 (action_id 200) NO_FRAME -- no frame_id == 200 exists.
    actions = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                action_id=1,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=25.0,
                end_y=34.0,
            ),
            dict(
                game_id="g",
                period_id=1,
                action_id=2,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=25.0,
                end_y=34.0,
            ),
            dict(
                game_id="g",
                period_id=1,
                action_id=3,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=70.0,
                end_y=34.0,
            ),
            dict(
                game_id="g",
                period_id=1,
                action_id=200,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=25.0,
                end_y=34.0,
            ),
        ]
    )
    frames = pd.DataFrame(
        [
            # frame 1 (scored): 3 teammates off the pass end + an opponent
            _keeper_row(1),
            _p(1, 11, 7, 30.0, 20.0),
            _p(1, 12, 7, 30.0, 48.0),
            _p(1, 13, 7, 50.0, 34.0),
            _p(1, 21, 8, 20.0, 34.0),
            _ball(1),
            # frame 2 (fov_cropped): keeper + teammates + opponent (dropped by the polygon, before options)
            _keeper_row(2),
            _p(2, 11, 7, 30.0, 20.0),
            _p(2, 12, 7, 30.0, 48.0),
            _p(2, 13, 7, 50.0, 34.0),
            _p(2, 21, 8, 20.0, 34.0),
            _ball(2),
            # frame 3 (too_few): keeper + ONE teammate + opponent -> chosen + 1 alt = 2 < min_options 3
            _keeper_row(3),
            _p(3, 11, 7, 40.0, 34.0),
            _p(3, 21, 8, 20.0, 34.0),
            _ball(3),
        ]
    )
    big = np.array([[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]])  # observes the keeper
    far = np.array([[100.0, 60.0], [105.0, 60.0], [105.0, 68.0], [100.0, 68.0]])  # far corner -> keeper unobserved
    visible_area = pd.DataFrame(
        {"action_id": [1, 2, 3], "polygon": [big, far, big]}  # action 200 needs none (dropped no_frame first)
    )

    # reachability pinned to 0.0 so this conservation test exercises the drop CLASSES (too_few_options from
    # candidate COUNT, no_frame, fov_cropped) independent of the reachability threshold -- the filter's
    # pruning is exercised separately in test_reconstruct.py; the shipped default (0.85) is asserted there.
    os_ = ReconstructedOptionSet(
        actions,
        frames,
        xpass=_FakeXPass(),
        params=GkDecisionParams(reachability_min_xpass=0.0),
        keeper_ids=[99],
        visible_area=visible_area,
    )
    samples, report = compute_gk_decision_value(os_)  # drops auto-pulled from os_.drop_counts()

    assert report.n_scored == 1
    assert report.n_too_few_options == 1
    assert report.n_no_frame == 1
    assert report.n_fov_cropped == 1
    assert report.n_no_unique_chosen == 0 and report.n_chosen_unvalued == 0
    # conservation: every one of the four decisions is accounted for exactly once
    assert (
        report.n_scored
        + report.n_too_few_options
        + report.n_no_unique_chosen
        + report.n_chosen_unvalued
        + report.n_no_frame
        + report.n_fov_cropped
        == report.n_decisions_in
        == 4
    )
    assert len(samples) == 1 and samples.iloc[0]["decision_id"] == 1
