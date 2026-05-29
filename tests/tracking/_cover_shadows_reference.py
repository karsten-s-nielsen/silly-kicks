"""Frozen, independent oracle for the cover-shadows leave-one-out (PR-S65).

!!! FROZEN at pre-PR-S65 behavior. Do NOT update this file to track production
!!! changes. Its entire value is being an INDEPENDENT reimplementation: keeping it
!!! "in sync" with production would make the exactness test circular and worthless.
!!! If production ever legitimately changes the value, that is a NEW spec and a NEW
!!! oracle -- never an edit here.

``_reference_lane_probabilities`` is a verbatim copy of the PRE-refactor
``_cover_shadows._compute_lane_probabilities`` (the sequential survival scan).
``_reference_max_single`` reproduces the ``detailed=False`` max_single computation
using that frozen scan with the man-marker set classified ONCE on the full frame
(fixed cast). It shares none of the new production helpers, so asserting
production == reference validates the helper extraction AND the vectorization
against independent code. See spec section 6.1.
"""

from __future__ import annotations

import numpy as np

from silly_kicks.tracking._cover_shadows import (
    CoverShadowParams,
    _classify_man_markers,
    ball_drag_time,
    player_tti,
)


def _reference_lane_probabilities(targets, defender_pos, defender_vel, attacker_pos, attacker_vel, *, params):
    """Frozen verbatim copy of pre-refactor _compute_lane_probabilities."""
    n_points = targets.shape[0]
    d_from_passer = np.linalg.norm(targets - targets[0:1], axis=1)
    t_ball = ball_drag_time(d_from_passer, params)
    tti_def = player_tti(defender_pos, defender_vel, targets, is_defender=True, params=params)
    tti_att = player_tti(attacker_pos, attacker_vel, targets, is_defender=False, params=params)
    s = np.sqrt(3.0) * params.sigma / np.pi

    def _p_int(tti_matrix):
        dt = t_ball[np.newaxis, :] - tti_matrix
        return 1.0 / (1.0 + np.exp(-dt / s))

    p_int_def = _p_int(tti_def)
    p_int_att = _p_int(tti_att)
    p_blocked = 0.0
    p_received = 0.0
    p_anyone_prior = 0.0
    for k in range(1, n_points):
        dt_k = t_ball[k] - t_ball[k - 1]
        if dt_k <= 0:
            continue
        p_ctrl = 1.0 - np.exp(-params.lambda_ctrl * dt_k)
        total_contrib_k = 0.0
        for j in range(len(defender_pos)):
            contrib = float(p_int_def[j, k]) * p_ctrl * (1.0 - p_anyone_prior)
            p_blocked += contrib
            total_contrib_k += contrib
        for j in range(len(attacker_pos)):
            contrib = float(p_int_att[j, k]) * p_ctrl * (1.0 - p_anyone_prior)
            p_received += contrib
            total_contrib_k += contrib
        p_anyone_prior = min(p_anyone_prior + total_contrib_k, 1.0)
    return p_blocked, p_received


def _reference_max_single(frame_data, passer_xy, attacking_team_id, xt, *, home_team_id):
    """Reference max_single_defender_blocking_score via frozen scan + fixed cast."""
    p = CoverShadowParams()
    players = frame_data[~frame_data["is_ball"].astype(bool)]
    attackers = players[players["team_id"] == attacking_team_id]
    attackers_outfield = attackers[~attackers["is_goalkeeper"].astype(bool)]
    ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
    ball_x = float(ball_rows.iloc[0]["x"])
    attacking_high = str(attacking_team_id) == str(home_team_id)
    dangerous = (
        attackers_outfield[attackers_outfield["x"] > ball_x]
        if attacking_high
        else attackers_outfield[attackers_outfield["x"] < ball_x]
    )
    if len(dangerous) == 0:
        return 0.0

    defenders_outfield = players[(players["team_id"] != attacking_team_id) & (~players["is_goalkeeper"].astype(bool))]
    goal_x_own = 105.0 if attacking_high else 0.0
    man_markers = _classify_man_markers(defenders_outfield, attackers, goal_x_own=goal_x_own, params=p)
    lane_blocker_ids = [pid for pid in defenders_outfield["player_id"] if pid not in man_markers]
    if not lane_blocker_ids:
        return 0.0

    xt_interp = xt.interpolator()
    passer = np.array(passer_xy, dtype=np.float64)
    att_pos = attackers[["x", "y"]].to_numpy(dtype=np.float64)
    att_vel = attackers[["vx", "vy"]].to_numpy(dtype=np.float64)

    kept = defenders_outfield[defenders_outfield["player_id"].isin(lane_blocker_ids)]
    full_pos = kept[["x", "y"]].to_numpy(dtype=np.float64)
    full_vel = kept[["vx", "vy"]].to_numpy(dtype=np.float64)
    kept_ids = kept["player_id"].to_numpy()

    max_def = 0.0
    for d_pid in lane_blocker_ids:
        keep_mask = kept_ids != d_pid
        score_d = 0.0
        for _, recv in dangerous.iterrows():
            recv_x = float(recv["x"])
            recv_y = float(recv["y"])
            recv_xt = float(xt_interp(np.array([recv_x]), np.array([recv_y]))[0, 0])
            receiver = np.array([recv_x, recv_y], dtype=np.float64)
            pass_vec = receiver - passer
            pass_dist = np.linalg.norm(pass_vec)
            if pass_dist < 1e-6:
                continue
            u = pass_vec / pass_dist
            u_perp = np.array([-u[1], u[0]])
            half_width = p.cone_width_factor * pass_dist / 2.0
            t = np.linspace(0.0, 1.0, p.n_sample_points)
            center = passer[np.newaxis, :] + t[:, np.newaxis] * pass_vec[np.newaxis, :]
            left = center + t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]
            right = center - t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]
            old_recv = 0.0
            new_recv = 0.0
            for lane in (center, left, right):
                _, base_rec = _reference_lane_probabilities(lane, full_pos, full_vel, att_pos, att_vel, params=p)
                _, loo_rec = _reference_lane_probabilities(
                    lane, full_pos[keep_mask], full_vel[keep_mask], att_pos, att_vel, params=p
                )
                old_recv += base_rec
                new_recv += loo_rec
            score_d += recv_xt * max(new_recv - old_recv, 0.0)
        max_def = max(max_def, score_d)
    return max_def
