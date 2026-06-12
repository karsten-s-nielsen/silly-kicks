"""ADR-028 durable guard: emitted action-LTR geometry is invariant under a physical
left/right mirror of the frame + action.

A situation that physically happens near the absolute-left goal must yield the SAME
action-LTR feature values as its mirror near the absolute-right goal. Any seam that
leaks frame (home-attacks-right) orientation into a per-action position output breaks
this invariant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_pre_shot_gk_context
from silly_kicks.tracking.features import add_defensive_line, add_ghost_gk, add_team_shape

HOME, AWAY = 1, 2
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
FL, FW = 105.0, 68.0


def _scenario():
    """Away team shoots toward x=105 (away action, frame is home-attacks-right)."""
    base = dict(
        game_id=1,
        period_id=1,
        frame_id=100,
        time_seconds=4.0,
        frame_rate=25.0,
        z=0.0,
        speed=0.0,
        speed_source="native",
        ball_state="alive",
        confidence=None,
        visibility=None,
        source_provider="synthetic",
        is_goalkeeper_source="native",
    )
    rows = [
        dict(
            player_id=1, team_id=HOME, is_ball=False, is_goalkeeper=True, x=4.0, y=31.0, team_attacking_direction="ltr"
        ),
        dict(
            player_id=50,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=True,
            x=101.0,
            y=37.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=11,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=20.0,
            y=30.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=12,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=24.0,
            y=44.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=13,
            team_id=HOME,
            is_ball=False,
            is_goalkeeper=False,
            x=16.0,
            y=20.0,
            team_attacking_direction="ltr",
        ),
        dict(
            player_id=61,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=15.0,
            y=30.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=62,
            team_id=AWAY,
            is_ball=False,
            is_goalkeeper=False,
            x=18.0,
            y=40.0,
            team_attacking_direction="rtl",
        ),
        dict(
            player_id=np.nan,
            team_id=np.nan,
            is_ball=True,
            is_goalkeeper=False,
            x=13.0,
            y=34.0,
            team_attacking_direction=None,
        ),
    ]
    frames = pd.DataFrame([{**base, **r} for r in rows])
    actions = pd.DataFrame(
        [
            dict(
                game_id=1,
                period_id=1,
                action_id=0,
                team_id=HOME,
                player_id=1.0,
                type_id=GOALKICK,
                result_id=1,
                start_x=5.0,
                start_y=31.0,
                end_x=40.0,
                end_y=34.0,
                time_seconds=3.6,
            ),
            dict(
                game_id=1,
                period_id=1,
                action_id=1,
                team_id=AWAY,
                player_id=61.0,
                type_id=SHOT,
                result_id=1,
                start_x=92.0,
                start_y=38.0,
                end_x=105.0,
                end_y=34.0,
                time_seconds=4.0,
            ),
        ]
    )
    return actions, frames


def _mirror(actions, frames):
    """Physical left/right mirror: flip all frame x/y and swap team_attacking_direction.

    The action coordinates are LEFT UNCHANGED on purpose: SPADL actions are already
    LTR-normalized (the acting team attacks x=105 regardless of physical orientation),
    so they are invariant under a physical mirror. Only the frame coordinate system and
    the per-team attacking direction flip. The emitted action-LTR geometry must therefore
    be identical between the two.
    """
    f = frames.copy()
    f["x"] = FL - f["x"]
    f["y"] = FW - f["y"]
    f["team_attacking_direction"] = f["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return actions.copy(), f


def _assert_invariant(base, mir, aid, cols, *, tol=1e-6):
    base = base.set_index("action_id")
    mir = mir.set_index("action_id")
    for col in cols:
        b, m = base.loc[aid, col], mir.loc[aid, col]
        assert (pd.isna(b) and pd.isna(m)) or abs(b - m) < tol, f"{col}: {b} vs {m}"


def test_pre_shot_gk_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_pre_shot_gk_context(a, frames=f)
    mir = add_pre_shot_gk_context(am, frames=fm)
    _assert_invariant(
        base,
        mir,
        1,
        [
            "pre_shot_gk_x",
            "pre_shot_gk_y",
            "pre_shot_gk_distance_to_goal",
            "pre_shot_gk_distance_to_shot",
        ],
    )


# NOTE on home_team_id in the mirrored call: the orientation-aware compute functions
# (compute_defensive_line) take home_team_id meaning "the team that attacks RIGHT in
# these frames" (= home for real convert_to_frames output, which is always
# home-attacks-right). After _mirror, the team attacking right is AWAY, so the mirrored
# call passes home_team_id=AWAY. The per-action re-projection itself reads
# team_attacking_direction and needs no home_team_id.


def test_defensive_line_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_defensive_line(a, f, home_team_id=HOME)
    mir = add_defensive_line(am, fm, home_team_id=AWAY)
    _assert_invariant(base, mir, 1, ["defensive_line_x", "back_line_high_x", "compactness_x"])


def test_team_shape_centroids_mirror_invariant():
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_team_shape(a, f, home_team_id=HOME)
    mir = add_team_shape(am, fm, home_team_id=AWAY)
    # ADR-028: compute_team_shape is now orientation-aware (deepest line nearest the
    # defended goal), so defensive_line_height is the team's true defensive line and is
    # mirror-invariant for BOTH teams alongside the centroids/spans.
    _assert_invariant(
        base,
        mir,
        1,
        [
            "team_shape_centroid_x_attacking",
            "team_shape_centroid_y_attacking",
            "team_shape_centroid_x_defending",
            "team_shape_centroid_y_defending",
            "team_shape_team_length_attacking",
            "team_shape_convex_hull_area_attacking",
            "team_shape_defensive_line_height_attacking",
            "team_shape_defensive_line_height_defending",
        ],
    )


def test_ghost_gk_mirror_invariant():
    # Pre-load the model ONCE and share it across both calls (avoids a double ~18s load)
    # and locks the asymmetric ghost transform (x uniform, y per-action flip).
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    model = GhostGkModel.from_variant("default")
    a, f = _scenario()
    am, fm = _mirror(a, f)
    base = add_ghost_gk(a, f, home_team_id=HOME, model=model)
    mir = add_ghost_gk(am, fm, home_team_id=AWAY, model=model)
    # Tolerance 0.5 m: the trained HGBR is not bit-symmetric under a coordinate mirror
    # (feature/KDE arithmetic + learned left/right asymmetry), so a sub-metre delta is
    # model noise, not an orientation leak. The DISCRIMINATING axis is x: a gross
    # orientation bug puts ghost_gk_x ~90 m away (goal-relative ~13 vs action-LTR ~101),
    # which this comfortably catches; near-goal-centre y is inherently low-discrimination.
    _assert_invariant(base, mir, 1, ["ghost_gk_x", "ghost_gk_y", "ghost_gk_density_spread"], tol=0.5)
