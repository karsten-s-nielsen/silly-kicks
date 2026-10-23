"""Task 2.6 — resolve_restart_geometry derived-row guard (spec section 9.1).

A derived (events-only) restart row never receives a frame-sampled or next-event coordinate. A derived
goal kick keeps the rule-point restart_prior; a derived free kick / corner / throw-in stays unresolved.
Tested frames-free: the next_event destination tier (a positional shift, no frames) is the one that would
otherwise fill a derived row, so excluding it proves the guard non-vacuously.
"""

import warnings

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as cfg
from silly_kicks.tracking._gk_geometry import resolve_restart_geometry


def _row(type_name, provenance, *, start=(np.nan, np.nan), end=(np.nan, np.nan)):
    return {
        "type_id": cfg.actiontype_id[type_name],
        "start_x": start[0],
        "start_y": start[1],
        "end_x": end[0],
        "end_y": end[1],
        "action_provenance": provenance,
    }


def test_derived_corner_stays_unresolved_next_event_skipped():
    acts = pd.DataFrame(
        [
            _row("corner_crossed", "derived"),
            _row("pass", "native", start=(50.0, 34.0), end=(60.0, 34.0)),
        ]
    )
    out = resolve_restart_geometry(acts)
    assert out.iloc[0]["start_coord_source"] == "unresolved"
    assert out.iloc[0]["end_coord_source"] == "unresolved"  # next_event skipped for the derived row


def test_native_corner_gets_next_event_dest_nonvacuous():
    # Guard OFF: the identical non-derived corner DOES take the next event's start as its destination.
    acts = pd.DataFrame(
        [
            _row("corner_crossed", "native"),
            _row("pass", "native", start=(50.0, 34.0), end=(60.0, 34.0)),
        ]
    )
    out = resolve_restart_geometry(acts)
    assert out.iloc[0]["end_coord_source"] == "next_event"


def test_derived_goalkick_gets_rule_point():
    acts = pd.DataFrame([_row("goalkick", "derived", end=(60.0, 34.0))])
    out = resolve_restart_geometry(acts)
    assert out.iloc[0]["start_coord_source"] == "restart_prior"


def test_derived_freekick_and_throwin_stay_unresolved():
    acts = pd.DataFrame(
        [
            _row("freekick_crossed", "derived", end=(60.0, 34.0)),
            _row("throw_in", "derived", end=(60.0, 34.0)),
        ]
    )
    out = resolve_restart_geometry(acts)
    assert out.iloc[0]["start_coord_source"] == "unresolved"
    assert out.iloc[1]["start_coord_source"] == "unresolved"


def test_no_action_provenance_column_is_unaffected():
    # A tracking provider with no derived rows carries no action_provenance -> the guard is a no-op and
    # a native goal kick keeps its rule-point.
    acts = pd.DataFrame(
        [{"type_id": cfg.actiontype_id["goalkick"], "start_x": np.nan, "start_y": np.nan, "end_x": 60.0, "end_y": 34.0}]
    )
    out = resolve_restart_geometry(acts)
    assert out.iloc[0]["start_coord_source"] == "restart_prior"


def test_tracking_tier_skipped_for_derived_with_frames_present():
    # WITH frames, a native corner's origin resolves via the tracking-ball tier; the derived corner's
    # does NOT (the tier-2b `& ~derived` guard). Non-vacuous: drop `& ~derived` and the derived row also
    # resolves, failing this (the B-10 surviving-mutation gap).
    from silly_kicks.spadl.utils import add_restart_coordinates

    acts = pd.DataFrame(
        dict(
            game_id=[9, 9],
            period_id=[1, 1],
            action_id=[0, 1],
            team_id=[1, 1],
            player_id=[10, 10],
            type_id=[cfg.actiontype_id["corner_crossed"], cfg.actiontype_id["corner_crossed"]],
            time_seconds=[5.0, 6.0],
            start_x=[np.nan, np.nan],
            start_y=[np.nan, np.nan],
            end_x=[np.nan, np.nan],
            end_y=[np.nan, np.nan],
            action_provenance=["native", "derived"],
        )
    )
    frames = pd.DataFrame(
        dict(
            game_id=[9, 9, 9, 9],
            period_id=[1, 1, 1, 1],
            frame_id=[100, 100, 200, 200],
            time_seconds=[5.0, 5.0, 6.0, 6.0],
            team_id=[0, 1, 0, 1],
            player_id=[-1, 10, -1, 10],
            is_goalkeeper=[False, False, False, False],
            is_ball=[True, False, True, False],
            team_attacking_direction=["ltr", "ltr", "ltr", "ltr"],
            x=[1.0, 20.0, 1.0, 20.0],  # ball in the corner region -> tracking_ball, no tripwire revert
            y=[1.0, 30.0, 1.0, 30.0],
            source_provider=["skillcorner"] * 4,
        )
    )
    with warnings.catch_warnings():
        # The native corner's midfield-ball origin trips the Law-region tripwire (incidental to this
        # test, which asserts the derived-vs-native source contrast, not the tripwire).
        warnings.simplefilter("ignore", UserWarning)
        out = add_restart_coordinates(acts, frames=frames).sort_values("action_id").reset_index(drop=True)
    native, derived = out.iloc[0], out.iloc[1]
    assert native["start_coord_source"] != "unresolved"  # tracking tier fired for the native corner
    assert derived["start_coord_source"] == "unresolved"  # guard: no frame-sampled origin for a derived row


def _corner_pair_frames():
    frames = pd.DataFrame(
        dict(
            game_id=[9, 9, 9, 9],
            period_id=[1, 1, 1, 1],
            frame_id=[100, 100, 200, 200],
            time_seconds=[5.0, 5.0, 6.0, 6.0],
            team_id=[0, 1, 0, 1],
            player_id=[-1, 10, -1, 10],
            is_goalkeeper=[False, False, False, False],
            is_ball=[True, False, True, False],
            team_attacking_direction=["ltr", "ltr", "ltr", "ltr"],
            x=[30.0, 20.0, 30.0, 20.0],
            y=[40.0, 30.0, 40.0, 30.0],
            source_provider=["skillcorner"] * 4,
        )
    )
    return frames


def test_destination_tracking_ball_skipped_for_derived():
    # The DERIVED corner is FIRST (so its next_event destination is available from the native corner that
    # follows); the guard still forces it to unresolved -- exercising both the next_event AND tracking_ball
    # `& ~derived` dest guards. The native corner is LAST (no next_event) -> its destination resolves via
    # the tracking-ball tier. Non-vacuous contrast.
    from silly_kicks.spadl.utils import add_restart_coordinates

    acts = pd.DataFrame(
        dict(
            game_id=[9, 9],
            period_id=[1, 1],
            action_id=[0, 1],
            team_id=[1, 1],
            player_id=[10, 10],
            type_id=[cfg.actiontype_id["corner_crossed"], cfg.actiontype_id["corner_crossed"]],
            time_seconds=[5.0, 6.0],
            start_x=[20.0, 20.0],  # resolved origin (not under test); END is NaN
            start_y=[30.0, 30.0],
            end_x=[np.nan, np.nan],
            end_y=[np.nan, np.nan],
            action_provenance=["derived", "native"],
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # incidental Law-region origin warning
        out = (
            add_restart_coordinates(acts, frames=_corner_pair_frames()).sort_values("action_id").reset_index(drop=True)
        )
    assert out.iloc[0]["end_coord_source"] == "unresolved"  # derived: next_event + tracking_ball both guarded
    assert out.iloc[1]["end_coord_source"] == "tracking_ball"  # native (last) resolves via tracking-ball


def test_tier2a_tracking_gk_skipped_for_derived_goalkick():
    # A goal kick with a detected keeper: the native goal kick's origin resolves via tracking_gk, the
    # derived goal kick's keeps the rule-point restart_prior (tier-2a `& ~derived`). Non-vacuous.
    from silly_kicks.spadl.utils import add_restart_coordinates

    frames = pd.DataFrame(
        dict(
            game_id=[9, 9, 9, 9],
            period_id=[1, 1, 1, 1],
            frame_id=[100, 100, 200, 200],
            time_seconds=[5.0, 5.0, 6.0, 6.0],
            team_id=[1, 0, 1, 0],
            player_id=[99, -1, 99, -1],
            is_goalkeeper=[True, False, True, False],  # keeper of the kicking team
            is_ball=[False, True, False, True],
            visibility=[True, True, True, True],  # detection-aware visibility (ADR-069)
            team_attacking_direction=["ltr", "ltr", "ltr", "ltr"],
            x=[9.0, 50.0, 9.0, 50.0],  # keeper in its own goal area (ltr attacker -> own box near x=0)
            y=[34.0, 34.0, 34.0, 34.0],
            source_provider=["skillcorner"] * 4,
        )
    )
    acts = pd.DataFrame(
        dict(
            game_id=[9, 9],
            period_id=[1, 1],
            action_id=[0, 1],
            team_id=[1, 1],
            player_id=[99, 99],
            type_id=[cfg.actiontype_id["goalkick"], cfg.actiontype_id["goalkick"]],
            time_seconds=[5.0, 6.0],
            start_x=[np.nan, np.nan],
            start_y=[np.nan, np.nan],
            end_x=[60.0, 60.0],
            end_y=[34.0, 34.0],
            action_provenance=["native", "derived"],
        )
    )
    out = add_restart_coordinates(acts, frames=frames).sort_values("action_id").reset_index(drop=True)
    assert out.iloc[0]["start_coord_source"] == "tracking_gk"  # tier 2a fired for the native goal kick
    assert out.iloc[1]["start_coord_source"] == "restart_prior"  # guard: derived keeps the rule-point


def test_distrust_origin_block_skips_derived_goalkick():
    # The goalkick DISTRUST-origin pre-cascade (_tracking_gk_xy_detected) must also skip derived rows
    # (B-14): native goal kick -> detected-keeper tracking_gk; derived goal kick -> rule-point. Direct
    # call with distrust_native_origin=True, which routes goal kicks through that block.
    frames = pd.DataFrame(
        dict(
            game_id=[9, 9, 9, 9],
            period_id=[1, 1, 1, 1],
            frame_id=[100, 100, 200, 200],
            time_seconds=[5.0, 5.0, 6.0, 6.0],
            team_id=[1, 0, 1, 0],
            player_id=[99, -1, 99, -1],
            is_goalkeeper=[True, False, True, False],
            is_ball=[False, True, False, True],
            visibility=[True, True, True, True],
            team_attacking_direction=["ltr", "ltr", "ltr", "ltr"],
            x=[9.0, 50.0, 9.0, 50.0],
            y=[34.0, 34.0, 34.0, 34.0],
            source_provider=["skillcorner"] * 4,
        )
    )
    acts = pd.DataFrame(
        dict(
            game_id=[9, 9],
            period_id=[1, 1],
            action_id=[0, 1],
            team_id=[1, 1],
            player_id=[99, 99],
            type_id=[cfg.actiontype_id["goalkick"], cfg.actiontype_id["goalkick"]],
            time_seconds=[5.0, 6.0],
            start_x=[np.nan, np.nan],
            start_y=[np.nan, np.nan],
            end_x=[60.0, 60.0],
            end_y=[34.0, 34.0],
            action_provenance=["native", "derived"],
        )
    )
    out = resolve_restart_geometry(acts, frames=frames, distrust_native_origin=True).sort_values("start_coord_source")
    srcs = set(out["start_coord_source"])
    assert srcs == {"tracking_gk", "restart_prior"}  # native detected-keeper; derived guarded to rule-point
