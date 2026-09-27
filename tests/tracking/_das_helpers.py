"""Shared builders for the native-DAS test suite (grown across plan Tasks 3-12).

Kept as plain functions (not pytest fixtures) so any test module can import them without fixture-plugin
wiring. Later tasks add ``run_engine`` / ``kernel_args`` / ``ghost_pair_from_golden`` here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._das_engine import compute_das
from silly_kicks.tracking._das_pack import pack_frames
from tests.tracking._das_golden import load_golden


def golden_frames(scene: str) -> pd.DataFrame:
    """The golden scene's frames with canonical dtypes applied."""
    return load_golden().frames_for(scene)


def single_frame(
    *, poss=1, direction=1.0, n1=11, n2=11, ball_xy=(52.5, 34.0), carrier=100, game=1, period=1, frame=0, seed=0
) -> pd.DataFrame:
    """One synthetic frame in raw silly-kicks coordinates (Int64 ids, float32 coords)."""
    rng = np.random.default_rng(seed)
    rows = []
    for team, base, n in ((1, 35.0, n1), (2, 70.0, n2)):
        for k in range(n):
            rows.append(
                dict(
                    game_id=game,
                    period_id=period,
                    frame_id=frame,
                    player_id=team * 100 + k,
                    team_id=team,
                    is_ball=False,
                    x=float(np.clip(base + rng.normal(0, 11), 1, 104)),
                    y=float(rng.uniform(2, 66)),
                    vx=float(rng.normal(0, 2)),
                    vy=float(rng.normal(0, 2)),
                    team_in_possession=poss,
                    ball_carrier_player_id=carrier,
                    dir=direction,
                )
            )
    rows.append(
        dict(
            game_id=game,
            period_id=period,
            frame_id=frame,
            player_id=pd.NA,
            team_id=pd.NA,
            is_ball=True,
            x=ball_xy[0],
            y=ball_xy[1],
            vx=0.0,
            vy=0.0,
            team_in_possession=poss,
            ball_carrier_player_id=carrier,
            dir=direction,
        )
    )
    df = pd.DataFrame(rows)
    df["player_id"] = df["player_id"].astype("Int64")
    df["team_id"] = df["team_id"].astype("Int64")
    df["team_in_possession"] = df["team_in_possession"].astype("Int64")
    df["ball_carrier_player_id"] = df["ball_carrier_player_id"].astype("Int64")
    for c in ("x", "y", "vx", "vy"):
        df[c] = df[c].astype("float32")
    return df


def run_engine(
    frames: pd.DataFrame, params, *, engine: Literal["auto", "numpy", "numba"] = "numpy", direction_col: str = "dir"
):
    """Pack ``frames`` (direction from ``direction_col``), run ``compute_das``, and return two tidy
    DataFrames aligned by key so a test can compare against ``GoldenFixture.reference_for``.

    Returns ``(team_df, player_df)`` with columns
    ``[game_id, period_id, frame_id, team_as, team_das]`` and
    ``[game_id, period_id, frame_id, player_id, player_as, player_das]`` (player_id canonical int).
    """
    packed = pack_frames(frames, attacking_direction_col=direction_col)
    res = compute_das(packed, params, engine=engine)
    keys = packed.keys
    team_df = pd.DataFrame(
        {
            "game_id": keys["game_id"].to_numpy(),
            "period_id": keys["period_id"].to_numpy(),
            "frame_id": keys["frame_id"].to_numpy(),
            "team_as": res.team_as,
            "team_das": res.team_das,
        }
    )
    counts = np.diff(packed.offsets)
    rep = keys.loc[keys.index.repeat(counts)].reset_index(drop=True)
    pids = frames["player_id"].to_numpy()[packed.p_input_pos]
    player_df = pd.DataFrame(
        {
            "game_id": rep["game_id"].to_numpy(),
            "period_id": rep["period_id"].to_numpy(),
            "frame_id": rep["frame_id"].to_numpy(),
            "player_id": [int(canonical_id(v)) for v in pids],  # type: ignore[arg-type]  # player rows: never NA
            "player_as": res.player_as,
            "player_das": res.player_das,
        }
    )
    return team_df, player_df


def engine_arrays(scene: str, params, *, engine: Literal["auto", "numpy", "numba"] = "numpy"):
    """``run_engine`` on a golden scene, aligned to ``reference_for(scene)`` ordering for direct compare."""
    g = load_golden()
    team_df, player_df = run_engine(g.frames_for(scene), params, engine=engine)
    team_df = team_df.sort_values(["game_id", "period_id", "frame_id"], kind="stable").reset_index(drop=True)
    player_df = player_df.sort_values(["game_id", "period_id", "frame_id", "player_id"], kind="stable").reset_index(
        drop=True
    )
    return {
        "team_as": team_df["team_as"].to_numpy(),
        "team_das": team_df["team_das"].to_numpy(),
        "player_as": player_df["player_as"].to_numpy(),
        "player_das": player_df["player_das"].to_numpy(),
    }


def kernel_args(dtype: type = np.float64):
    """Positional argument tuple for the numba serial kernel, built from golden S01.

    ``dtype`` casts the kinematic arrays (px/py/pvx/pvy) -- pass float32 to exercise the
    float64-signature rejection.
    """
    from silly_kicks.tracking._das_pack import Reason, pack_frames
    from silly_kicks.tracking._das_params import DAS_PARAMS, simulation_grids

    packed = pack_frames(golden_frames("S01"), attacking_direction_col="dir")
    g = simulation_grids(DAS_PARAMS)
    p = DAS_PARAMS
    n, n_rows = packed.n_frames, len(packed.px)
    ok = np.flatnonzero(packed.reason == Reason.OK).astype(np.int64)
    return (
        ok,
        packed.offsets.astype(np.int64),
        packed.px.astype(dtype),
        packed.py.astype(dtype),
        packed.pvx.astype(dtype),
        packed.pvy.astype(dtype),
        packed.p_attacking.astype(np.bool_),
        packed.p_is_passer.astype(np.bool_),
        np.ascontiguousarray(packed.ball_xy[:, 0]),
        np.ascontiguousarray(packed.ball_xy[:, 1]),
        packed.direction.astype(np.float64),
        g.cos_phi,
        g.sin_phi,
        g.v0,
        g.d,
        g.t_ball,
        g.dt0,
        g.rate_divisor,
        g.dr,
        g.d_area,
        p.b0,
        p.b1,
        p.player_velocity,
        p.inertial_seconds,
        p.tol_distance,
        p.use_max,
        p.v_max,
        p.a_max,
        p.factor2,
        p.normalize,
        p.respect_offside,
        p.exclude_passer,
        p.danger_weight,
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n_rows, np.nan),
        np.full(n_rows, np.nan),
    )


def ghost_pair_from_golden(scene: str = "S01", *, keeper_shift_m: float = 3.0):
    """(actual, ghost) frames for a scene: the ghost moves the defending keeper towards midfield.

    The defending team is the team NOT in possession; the keeper is that team's deepest player (lowest
    x when the possession team attacks +x). Only that one row's x moves; every other column is identical.
    """
    frames = golden_frames(scene)
    frames = frames.copy()
    frames["is_goalkeeper"] = False
    ghost = frames.copy()
    # possession team attacks +x (dir=+1 in S01); defenders = team 2, keeper = deepest (max x).
    players = frames[~frames["is_ball"]]
    defenders = players[players["team_id"] == 2]
    keeper_pos = defenders.loc[defenders["x"].astype(float).idxmax()]
    kid = keeper_pos["player_id"]
    mask = (~ghost["is_ball"]) & (ghost["player_id"] == kid)
    ghost.loc[mask, "x"] = (ghost.loc[mask, "x"].astype(float) - keeper_shift_m).astype("float32")
    return frames, ghost
