"""Shared TF-30 cover-shadow test inputs.

Lives beside ``_provider_inputs.py`` because BOTH ``tests/invariants/`` and ``tests/tracking/``
consume it, and a fixture defined in a test MODULE is invisible to a sibling directory
(``tests/invariants/`` has no ``conftest.py``). Cross-directory import of this package is the
established pattern here -- ``_provider_inputs`` is already imported from ``tests/invariants/`` and
``tests/calibration/``.

A shared ``tests/conftest.py`` fixture would NOT do: at ``scope="module"`` it builds once per
consuming module -- two builds, not one -- and it would mean widening the root ``fitted_xt`` that
``tests/vaep/``, ``tests/tracking/`` and ``tests/invariants/`` all share. Memoizing here makes the
build session-wide: once, regardless of how many modules consume it.

CALLERS MUST ``.copy()`` BEFORE MUTATING. The thin fixtures in each test file do that; the copy is
what ``test_per_test_fixture_is_a_copy`` pins.
"""

from __future__ import annotations

import functools

import numpy as np

from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions


@functools.cache
def fitted_xt():
    """Module-local xT, byte-identical to the conftest fixtures.

    Deliberately NOT the root ``tests/conftest.py`` fixture: this module must not depend on the
    scope of a fixture shared with ``tests/vaep/`` and ``tests/tracking/``.

    It does NOT ``.fit()``. Fitting on the ~10 synthetic fixture actions yields a degenerate
    all-zero grid, every counterfactual becomes a no-op, and the resulting all-zero threat reads
    exactly like fixture inadequacy. If a cover-shadow test shows all-zero threat, suspect the xT
    grid before the fixture.
    """
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


@functools.cache
def prepared_frames_and_actions():
    """``(frames, actions, home_team_id)`` -- the shared expensive chain, once per session."""
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames
    from silly_kicks.tracking.utils import play_left_to_right

    frames = load_provider_frames("sportec")
    frames = smooth_frames(frames)
    frames = derive_velocities(frames)
    home_team_id = frames[~frames["team_id"].isna()]["team_id"].iloc[0]
    frames = play_left_to_right(frames, home_team_id=home_team_id)
    return frames, synthesize_actions(frames), home_team_id


@functools.cache
def cover_shadow_result():
    """``add_cover_shadows`` output, once per session. COPY BEFORE MUTATING."""
    from silly_kicks.tracking.features import add_cover_shadows

    frames, actions, home_team_id = prepared_frames_and_actions()
    return add_cover_shadows(actions, frames, fitted_xt(), home_team_id=home_team_id)


@functools.cache
def cover_shadow_result_detailed():
    """``add_cover_shadows(detailed=True)`` output, once per session. COPY BEFORE MUTATING.

    A SEPARATE memoized build rather than a parameter on :func:`cover_shadow_result`, because the
    two are not interchangeable and callers should have to say which they mean:
    ``max_single_defender_player_id`` is populated ONLY here. Under the default ``detailed=False``
    it is entirely NA by design -- the cheap path's argmax measured 0.157 agreement with the exact
    path, so it deliberately names nobody.

    Costs ~2.3-3.2x the cheap build (measured: 39-42 ms/action vs 98-125 ms/action), which is why
    it is memoized separately instead of replacing the default one.
    """
    from silly_kicks.tracking.features import add_cover_shadows

    frames, actions, home_team_id = prepared_frames_and_actions()
    return add_cover_shadows(actions, frames, fitted_xt(), home_team_id=home_team_id, detailed=True)


@functools.cache
def cover_shadow_raw():
    """Per-action ``BlockingScoreResult``, UNCLAMPED. Memoized; treat as read-only.

    Mirrors ``add_cover_shadows``' own action->frame resolution (``features.py``) so the tests
    score exactly the frames production does.

    ``link_actions_to_frames`` lives in ``tracking/utils.py`` and is re-exported from
    ``silly_kicks.tracking``; there is no ``silly_kicks.tracking.linkage`` module.
    """
    import pandas as pd

    from silly_kicks.tracking import link_actions_to_frames
    from silly_kicks.tracking._cover_shadows import compute_blocking_score

    frames, actions, home_team_id = prepared_frames_and_actions()
    xt = fitted_xt()

    pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    rows = []
    for _idx, row in actions.iterrows():
        aid, tid = row["action_id"], row["team_id"]
        if pd.isna(tid) or aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue
        try:
            frame_data = frame_groups.get_group((row["period_id"], int(float(fid_raw))))  # type: ignore[arg-type]
        except KeyError:
            continue
        res = compute_blocking_score(frame_data, tid, xt, home_team_id=home_team_id)
        rows.append((aid, frame_data, tid, res))
    return {"rows": rows, "home_team_id": home_team_id, "xt": xt}


def iter_scoreable():
    """Yield ``(frame_data, passer_xy, attacking_team_id)`` per resolvable action."""
    import pandas as pd

    from silly_kicks.tracking import link_actions_to_frames

    frames, actions, _home = prepared_frames_and_actions()
    pointers, _ = link_actions_to_frames(actions, frames)
    pointer_lookup = pointers.set_index("action_id")
    frame_groups = frames.groupby(["period_id", "frame_id"])

    for _idx, row in actions.iterrows():
        aid, tid = row["action_id"], row["team_id"]
        if pd.isna(tid) or aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]
        if pd.isna(fid_raw):
            continue
        try:
            frame_data = frame_groups.get_group((row["period_id"], int(float(fid_raw))))  # type: ignore[arg-type]
        except KeyError:
            continue
        yield frame_data, (float(row["start_x"]), float(row["start_y"])), tid


def lane_arrays(frame_data, attacking_team_id, home_team_id):
    """The cheap path's array inputs, mirroring ``_cover_shadows.py:1090-1097`` exactly.

    Returns ``(lb_pos, lb_vel, att_pos, att_vel, dangerous, cs_params)`` or ``None`` when the
    action is not scoreable by the production path (no velocities, no ball, no dangerous
    receiver, no lane blocker) -- the same four early returns
    ``_compute_cover_shadow_dict`` takes at ``:973``, ``:981``, ``:994`` and ``:1030``.

    Mirrored HERE rather than in the test so the construction has one home. A second copy in a
    test file is a copy that drifts silently when the production path is refactored.
    """
    import numpy as np
    import pandas as pd

    import silly_kicks.tracking._cover_shadows as cs
    from silly_kicks.id_compat import ids_match, same_id

    if "vx" not in frame_data.columns or "vy" not in frame_data.columns:
        return None

    players = frame_data[~frame_data["is_ball"].astype(bool)]
    attackers = players[ids_match(players["team_id"], attacking_team_id)]
    attackers_outfield = attackers[~attackers["is_goalkeeper"].astype(bool)]

    ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
    if ball_rows.empty or pd.isna(ball_rows.iloc[0]["x"]):
        return None
    ball_x = float(ball_rows.iloc[0]["x"])

    if same_id(attacking_team_id, home_team_id):
        dangerous = attackers_outfield[attackers_outfield["x"] > ball_x]
    else:
        dangerous = attackers_outfield[attackers_outfield["x"] < ball_x]
    if len(dangerous) == 0:
        return None

    blocker_ids = lane_blocker_ids(frame_data, attacking_team_id, home_team_id)
    if not blocker_ids:
        return None

    defenders_outfield = players[
        (~ids_match(players["team_id"], attacking_team_id)) & (~players["is_goalkeeper"].astype(bool))
    ]
    kept = defenders_outfield[defenders_outfield["player_id"].isin(blocker_ids)]
    return (
        kept[["x", "y"]].to_numpy(dtype=np.float64),
        kept[["vx", "vy"]].to_numpy(dtype=np.float64),
        attackers[["x", "y"]].to_numpy(dtype=np.float64),
        attackers[["vx", "vy"]].to_numpy(dtype=np.float64),
        dangerous,
        cs.CoverShadowParams(),
    )


def lane_blocker_ids(frame_data, attacking_team_id, home_team_id):
    """The candidate set the production path scores.

    Mirrors ``_cover_shadows.py``'s own construction exactly. Note the real signature:
    ``_classify_man_markers(defenders, attackers, *, goal_x_own, params)``.
    """
    import silly_kicks.tracking._cover_shadows as cs
    from silly_kicks.id_compat import ids_match, same_id

    players = frame_data[~frame_data["is_ball"].astype(bool)]
    defenders_outfield = players[
        (~ids_match(players["team_id"], attacking_team_id)) & (~players["is_goalkeeper"].astype(bool))
    ]
    attackers = players[ids_match(players["team_id"], attacking_team_id)]
    goal_x_own = 105.0 if same_id(attacking_team_id, home_team_id) else 0.0
    man_markers = cs._classify_man_markers(
        defenders_outfield, attackers, goal_x_own=goal_x_own, params=cs.CoverShadowParams()
    )
    return [pid for pid in defenders_outfield["player_id"] if pid not in man_markers]
