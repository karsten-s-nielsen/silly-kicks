"""Synthetic SPADL cohorts for xtgk tests. No real data — CI-safe.

Pressure fixtures use THREE well-separated bands (0.1 / 0.5 / 0.9) so all three terciles
populate. Deep goal-kicks are SPREAD across several deep cells so the occupied-cell gate has
>1 populated deep cell. Each possession routes through the BUILD-UP band (grid xi=4) so the
gate's cross-check comparison is not vacuously zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig

PASS = spadlconfig.actiontype_id["pass"]
DRIBBLE = spadlconfig.actiontype_id["dribble"]
CROSS = spadlconfig.actiontype_id["cross"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
THROW_IN = spadlconfig.actiontype_id["throw_in"]
SHOT = spadlconfig.actiontype_id["shot"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]

# three pressure bands -> terciles {1,2,3}; xg decreases with pressure (a real gradient)
BANDS = ((0.1, 0.5), (0.5, 0.25), (0.9, 0.05))  # (pressure, shot_xg)
DEEP_YS = (12.0, 24.0, 34.0, 44.0, 56.0)  # spread deep goal-kick origins across cells


def _row(
    action_id,
    type_id,
    result_id,
    sx,
    sy,
    ex,
    ey,
    *,
    game_id=1,
    period_id=1,
    team_id=10,
    player_id=100,
    possession_id=0,
    time_seconds=0.0,
    xg=np.nan,
    pressure=0.5,
):
    return dict(
        game_id=game_id,
        period_id=period_id,
        action_id=action_id,
        time_seconds=time_seconds,
        team_id=team_id,
        player_id=player_id,
        type_id=type_id,
        result_id=result_id,
        bodypart_id=0,
        start_x=sx,
        start_y=sy,
        end_x=ex,
        end_y=ey,
        possession_id=possession_id,
        xg=xg,
        pressure=pressure,
    )


def make_cohort(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)


def build_up_to_shot_possession(
    possession_id,
    *,
    pressure,
    xg,
    deep_x=3.0,
    deep_y=34.0,
    buildup_x=30.0,
    mid_x=55.0,
    shot_x=100.0,
    chain_y=34.0,
    base_action_id=0,
    game_id=1,
):
    """deep GOALKICK -> BUILD-UP pass -> mid pass -> forward pass -> shot(xg). Attack-LTR.

    The goal-kick lands at buildup_x=30 (grid xi=4, in BUILD_UP_CELLS xi 2..6) and the next pass
    STARTS there, so the build-up band is populated with real support and value — WITHOUT this the
    gate's cross-check comparison (graded on the build-up band) is vacuously 0==0.

    ``chain_y`` is the y of the build-up→shot chain (default 34, the pitch midline). The
    equivariance fixture overrides it to an off-midline value: on a 12-cell grid y=34 rounds into
    cell 6 (not the symmetric 5/6 boundary), so a y-reflection golden needs off-midline data.
    """
    a = base_action_id
    return [
        _row(
            a + 0,
            GOALKICK,
            SUCCESS,
            deep_x,
            deep_y,
            buildup_x,
            chain_y,
            game_id=game_id,
            possession_id=possession_id,
            pressure=pressure,
            time_seconds=a + 0.0,
        ),
        _row(
            a + 1,
            PASS,
            SUCCESS,
            buildup_x,
            chain_y,
            mid_x,
            chain_y,
            game_id=game_id,
            possession_id=possession_id,
            pressure=pressure,
            time_seconds=a + 1.0,
        ),
        _row(
            a + 2,
            PASS,
            SUCCESS,
            mid_x,
            chain_y,
            80.0,
            chain_y,
            game_id=game_id,
            possession_id=possession_id,
            pressure=pressure,
            time_seconds=a + 2.0,
        ),
        _row(
            a + 3,
            PASS,
            SUCCESS,
            80.0,
            chain_y,
            shot_x,
            chain_y,
            game_id=game_id,
            possession_id=possession_id,
            pressure=pressure,
            time_seconds=a + 3.0,
        ),
        _row(
            a + 4,
            SHOT,
            FAIL,
            shot_x,
            chain_y,
            105.0,
            chain_y,
            game_id=game_id,
            possession_id=possession_id,
            pressure=pressure,
            xg=xg,
            time_seconds=a + 4.0,
        ),
    ]


def three_band_cohort(n_per_band=40) -> pd.DataFrame:
    """Honest cohort across 3 pressure bands and several deep cells."""
    rows: list[dict] = []
    pid = 0
    for k in range(n_per_band):
        for bi, (pressure, xg) in enumerate(BANDS):
            deep_y = DEEP_YS[(k + bi) % len(DEEP_YS)]
            base = 1000 * pid
            rows += build_up_to_shot_possession(pid, pressure=pressure, xg=xg, deep_y=deep_y, base_action_id=base)
            pid += 1
    return make_cohort(rows)


def offmidline_cohort(n_per_band=40) -> pd.DataFrame:
    """Like three_band_cohort but with the build-up chain at y=30 (cell 5) and deep goal-kicks on
    off-midline cells that reflect cleanly (12<->56 cell 2<->9, 24<->44 cell 4<->7). Used by the
    y-reflection equivariance golden so the row-reversal is exact (avoids the y=34 cell-6 artifact).
    """
    off_ys = (12.0, 24.0, 44.0, 56.0)
    rows: list[dict] = []
    pid = 0
    for k in range(n_per_band):
        for bi, (pressure, xg) in enumerate(BANDS):
            deep_y = off_ys[(k + bi) % len(off_ys)]
            base = 1000 * pid
            rows += build_up_to_shot_possession(
                pid, pressure=pressure, xg=xg, deep_y=deep_y, chain_y=30.0, base_action_id=base
            )
            pid += 1
    return make_cohort(rows)


def flat_no_shot_cohort(n_per_band=40) -> pd.DataFrame:
    """Negative control: deep possessions that NEVER reach a shot -> deep V ~ 0, flat."""
    rows: list[dict] = []
    pid = 0
    for k in range(n_per_band):
        for pressure, _xg in BANDS:
            base = 1000 * pid
            rows += [
                _row(
                    base,
                    GOALKICK,
                    SUCCESS,
                    3.0,
                    DEEP_YS[k % len(DEEP_YS)],
                    40.0,
                    34.0,
                    possession_id=pid,
                    pressure=pressure,
                ),
                _row(base + 1, PASS, SUCCESS, 40.0, 34.0, 55.0, 34.0, possession_id=pid, pressure=pressure),
            ]
            pid += 1
    return make_cohort(rows)


def deep_low_rest_high_cohort(n_per_cell=20) -> pd.DataFrame:
    """Deep zone globally LOW pressure (goal-kicks ~0.02..0.18); outfield build-up/shots HIGH
    (~0.6..0.95). Under GLOBAL terciles the deep high tercile is starved -> the ladder STOPs on
    rung 1; under ZONE-CONDITIONAL terciles the deep band's own 0.02..0.18 spread splits into thirds
    -> the two deep cells (deep_y in {24,44}, both xi=0) populate all three terciles -> rung 2 fires."""
    rows: list[dict] = []
    pid = 0
    for dy in (24.0, 44.0):
        for k in range(n_per_cell):
            base = 1000 * pid
            low_p = 0.02 + 0.16 * (k / max(n_per_cell - 1, 1))  # deep-band spread 0.02..0.18
            hi_p = 0.6 + 0.35 * ((k % 3) / 2)  # outfield 0.60/0.775/0.95
            xg = 0.4 - 0.3 * (k / max(n_per_cell - 1, 1))  # xg falls as deep pressure rises
            rows += [
                _row(
                    base + 0,
                    GOALKICK,
                    SUCCESS,
                    3.0,
                    dy,
                    30.0,
                    34.0,
                    possession_id=pid,
                    pressure=low_p,
                    time_seconds=base + 0.0,
                ),
                _row(
                    base + 1,
                    PASS,
                    SUCCESS,
                    30.0,
                    34.0,
                    55.0,
                    34.0,
                    possession_id=pid,
                    pressure=hi_p,
                    time_seconds=base + 1.0,
                ),
                _row(
                    base + 2,
                    PASS,
                    SUCCESS,
                    55.0,
                    34.0,
                    80.0,
                    34.0,
                    possession_id=pid,
                    pressure=hi_p,
                    time_seconds=base + 2.0,
                ),
                _row(
                    base + 3,
                    PASS,
                    SUCCESS,
                    80.0,
                    34.0,
                    100.0,
                    34.0,
                    possession_id=pid,
                    pressure=hi_p,
                    time_seconds=base + 3.0,
                ),
                _row(
                    base + 4,
                    SHOT,
                    FAIL,
                    100.0,
                    34.0,
                    105.0,
                    34.0,
                    possession_id=pid,
                    pressure=hi_p,
                    xg=xg,
                    time_seconds=base + 4.0,
                ),
            ]
            pid += 1
    return make_cohort(rows)


def mixed_shot_and_shotless_cohort(n_per_band=40) -> pd.DataFrame:
    """Both classes for the construct-validity target: shot-reaching possessions (three_band, y=1)
    AND shotless possessions (flat_no_shot, y=0), in one game with distinct possession/action ids."""
    shot = three_band_cohort(n_per_band=n_per_band)
    noshot = flat_no_shot_cohort(n_per_band=n_per_band).copy()
    noshot["possession_id"] = noshot["possession_id"] + 100_000
    noshot["action_id"] = noshot["action_id"] + 10_000_000
    return (
        pd.concat([shot, noshot], ignore_index=True)
        .sort_values(["game_id", "period_id", "action_id"])
        .reset_index(drop=True)
    )


def mirror_y(actions: pd.DataFrame) -> pd.DataFrame:
    """Vertical reflection y->68-y ONLY. Attack direction (x) is PRESERVED -> still attack-LTR."""
    out = actions.copy()
    out["start_y"] = spadlconfig.field_width - actions["start_y"]
    out["end_y"] = spadlconfig.field_width - actions["end_y"]
    return out


def mirror_x(actions: pd.DataFrame) -> pd.DataFrame:
    """Horizontal reflection x->105-x ONLY. REVERSES attack direction -> NOT attack-LTR."""
    out = actions.copy()
    out["start_x"] = spadlconfig.field_length - actions["start_x"]
    out["end_x"] = spadlconfig.field_length - actions["end_x"]
    return out
