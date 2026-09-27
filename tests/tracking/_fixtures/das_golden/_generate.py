"""Generate the committed DAS golden fixture from the ``accessible-space`` 2.0.15 ORACLE.

This is the parity oracle for the native DAS engine (spec 2026-09-26-das-native-design.md, §7.1;
plan Task 1). It builds deterministic synthetic scenes and records what ``accessible-space`` 2.0.15
produces for each, so CI can assert the native engine reproduces the library in ``reference``
quadrature WITHOUT installing the library. It requires the dev-only ``das-reference`` extra
(``accessible-space==2.0.15``); regular CI never runs it.

Design contract:
* Coordinates and velocities are snapped to values EXACTLY representable in float32
  (``float(np.float32(v))``), so a native engine reading float32 STORAGE and upcasting to float64
  sees identical bits -- the fixture carries NO storage-rounding gap (mirrors the ADR-106 golden).
* Stored frames use the NATIVE convention: the ball is a row with ``is_ball == True`` and a NA
  ``player_id`` / ``team_id`` (no ``"ball"`` sentinel). The library, which needs the sentinel, is
  fed a private copy where the ball's ``player_id`` becomes ``"ball"`` and ids become object strings.
* Normal scenes are simulated one ``(game_id, period_id)`` at a time with the scene's KNOWN
  direction pinned through ``attacking_direction_col`` -- so the library's ``frame_id``-only keying
  cannot conflate periods/games and parity is measured with the same direction the native engine
  uses. Divergence scenes are run in the defect-exhibiting way and whatever the library does (a
  value, or a raise) is recorded.
* Serialisation is deterministic (rows sorted by key, ``%.17g`` floats, ``\\n`` newlines, UTF-8 no
  BOM, sorted JSON keys) so the generator reproduces the committed bytes under the pinned deps.

Run:  ``.venv/Scripts/python tests/tracking/_fixtures/das_golden/_generate.py --out <dir>``
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# Pitch: raw silly-kicks [0,105]x[0,68]; the library works in centred [-52.5,52.5]x[-34,34].
_X_OFF = 52.5
_Y_OFF = 34.0

# Stored frame columns, in this order.
_FRAME_COLS = [
    "scene_id",
    "game_id",
    "period_id",
    "frame_id",
    "player_id",
    "team_id",
    "is_ball",
    "x",
    "y",
    "vx",
    "vy",
    "team_in_possession",
    "dir",
]
_PASS_COLS = [
    "scene_id",
    "pass_index",
    "game_id",
    "period_id",
    "frame_id",
    "player_id",
    "team_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]


def _snap(v: float) -> float:
    """The float32-representable value of ``v`` (so float64 compute has no storage gap)."""
    return float(np.float32(v))


def _canonical_msg(msg: str) -> str:
    """Sort the contents of every ``{...}`` set-repr so a captured message is deterministic.

    The library embeds Python ``set`` reprs (e.g. ``{'t1', None, 't2'}``) in its error text; set
    iteration order is hash-seed-dependent, which would break the byte-reproducible fixture. Sorting
    the comma-separated members canonicalises them without losing the evidence.
    """

    def _sort_group(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner:
            return "{}"
        parts = [p.strip() for p in inner.split(",")]
        return "{" + ", ".join(sorted(parts)) + "}"

    return re.sub(r"\{([^{}]*)\}", _sort_group, msg)


def _row(scene, g, p, f, pid, tid, is_ball, x, y, vx, vy, poss, direction):
    return dict(
        scene_id=scene,
        game_id=g,
        period_id=p,
        frame_id=f,
        player_id=pid,
        team_id=tid,
        is_ball=bool(is_ball),
        x=_snap(x),
        y=_snap(y),
        vx=_snap(vx),
        vy=_snap(vy),
        team_in_possession=poss,
        dir=float(direction),
    )


def _frame(scene, g, p, f, *, poss, direction, seed, n1=11, n2=11, ball_xy=(52.5, 34.0), mutate=None):
    """One frame: n1 players of team 1 (base x=35), n2 of team 2 (base x=70), plus the ball."""
    rng = np.random.default_rng(seed)
    rows = []
    for team, base, n in ((1, 35.0, n1), (2, 70.0, n2)):
        for k in range(n):
            rows.append(
                _row(
                    scene,
                    g,
                    p,
                    f,
                    team * 100 + k,
                    team,
                    False,
                    float(np.clip(base + rng.normal(0, 11), 1, 104)),
                    float(rng.uniform(2, 66)),
                    float(rng.normal(0, 2)),
                    float(rng.normal(0, 2)),
                    poss,
                    direction,
                )
            )
    rows.append(_row(scene, g, p, f, None, None, True, ball_xy[0], ball_xy[1], 0.0, 0.0, poss, direction))
    if mutate is not None:
        rows = mutate(rows)
    return rows


# --------------------------------------------------------------------------------------------------
# Scene catalogue
# --------------------------------------------------------------------------------------------------


def _normal_scenes():
    """Scenes simulated per (game, period) with a pinned direction. Returns {scene_id: rows}."""
    scenes: dict[str, list] = {}

    def build(scene, specs):
        rows = []
        for spec in specs:
            rows += _frame(scene, **spec)
        scenes[scene] = rows

    # S01 home (team1) attacks +x.
    build("S01", [dict(g=1, p=1, f=f, poss=1, direction=+1, seed=100 + f) for f in range(3)])
    # S02 team2 attacks -x.
    build("S02", [dict(g=1, p=1, f=f, poss=2, direction=-1, seed=200 + f) for f in range(3)])
    # S03 two periods, direction flips, disjoint frame ids, possession alternates.
    build(
        "S03",
        [dict(g=1, p=1, f=f, poss=1, direction=+1, seed=300 + f) for f in range(3)]
        + [dict(g=1, p=2, f=100 + f, poss=2, direction=+1, seed=310 + f) for f in range(3)],
    )
    # S04 two games, disjoint frame ids.
    build(
        "S04",
        [dict(g=1, p=1, f=f, poss=1, direction=+1, seed=400 + f) for f in range(2)]
        + [dict(g=2, p=1, f=500 + f, poss=1, direction=+1, seed=420 + f) for f in range(2)],
    )

    # S05 active offside: place a team1 attacker beyond the 2nd-last team2 defender, ahead of ball.
    def offside_mut(rows):
        for r in rows:
            if r["team_id"] == 2 and not r["is_ball"]:
                r["x"] = _snap(30.0)  # push all defenders deep so the offside line is low
        # one attacker (player 100) far upfield, in the opponent half, ahead of the line
        for r in rows:
            if r["player_id"] == 100:
                r["x"] = _snap(90.0)
                r["y"] = _snap(40.0)
            if r["is_ball"]:
                r["x"] = _snap(50.0)
        return rows

    build("S05", [dict(g=1, p=1, f=f, poss=1, direction=+1, seed=500 + f, mutate=offside_mut) for f in range(2)])

    # S06 carrier itself beyond the offside line (exclusion matters). carrier = player 100.
    def carrier_offside_mut(rows):
        for r in rows:
            if r["team_id"] == 2 and not r["is_ball"]:
                r["x"] = _snap(30.0)
            if r["player_id"] == 100:
                r["x"] = _snap(92.0)
                r["y"] = _snap(34.0)
            if r["is_ball"]:
                r["x"] = _snap(50.0)
        return rows

    build(
        "S06", [dict(g=1, p=1, f=f, poss=1, direction=+1, seed=600 + f, mutate=carrier_offside_mut) for f in range(2)]
    )

    # S07 ball off the pitch (raw y < 0 -> centred y < -34).
    build("S07", [dict(g=1, p=1, f=f, poss=1, direction=+1, seed=700 + f, ball_xy=(50.0, -0.5)) for f in range(2)])

    # S08 one player with NaN x/y.
    def nan_pos_mut(rows):
        for r in rows:
            if r["player_id"] == 205:
                r["x"] = float("nan")
                r["y"] = float("nan")
        return rows

    build("S08", [dict(g=1, p=1, f=f, poss=1, direction=+1, seed=800 + f, mutate=nan_pos_mut) for f in range(2)])

    # S09 one player NaN vx/vy; frame 0 has it near the trajectory (within tol), frame 1 far.
    def nan_vel_near(rows):
        for r in rows:
            if r["player_id"] == 105:
                r["x"] = _snap(52.0)
                r["y"] = _snap(34.0)
                r["vx"] = float("nan")
                r["vy"] = float("nan")
        return rows

    def nan_vel_far(rows):
        for r in rows:
            if r["player_id"] == 105:
                r["x"] = _snap(10.0)
                r["y"] = _snap(5.0)
                r["vx"] = float("nan")
                r["vy"] = float("nan")
        return rows

    scenes["S09"] = _frame("S09", 1, 1, 0, poss=1, direction=+1, seed=900, mutate=nan_vel_near) + _frame(
        "S09", 1, 1, 1, poss=1, direction=+1, seed=901, mutate=nan_vel_far
    )

    # S10 ragged substitutes: union of 26 players, 20-22 present per frame.
    def s10():
        rows = []
        rng = np.random.default_rng(1000)
        rosters = [
            [100 + k for k in range(11)] + [200 + k for k in range(10)],  # 21
            [100 + k for k in range(10)] + [211] + [200 + k for k in range(11)],  # 22
            [100 + k for k in range(1, 11)] + [200 + k for k in range(10)],  # 20
        ]
        for f, roster in enumerate(rosters):
            for pid in roster:
                team = 1 if pid < 200 else 2
                base = 35.0 if team == 1 else 70.0
                rows.append(
                    _row(
                        "S10",
                        1,
                        1,
                        f,
                        pid,
                        team,
                        False,
                        float(np.clip(base + rng.normal(0, 11), 1, 104)),
                        float(rng.uniform(2, 66)),
                        float(rng.normal(0, 2)),
                        float(rng.normal(0, 2)),
                        1,
                        +1,
                    )
                )
            rows.append(_row("S10", 1, 1, f, None, None, True, 52.5, 34.0, 0.0, 0.0, 1, +1))
        return rows

    scenes["S10"] = s10()

    # S11 possession switches between frames.
    scenes["S11"] = _frame("S11", 1, 1, 0, poss=1, direction=+1, seed=1100) + _frame(
        "S11", 1, 1, 1, poss=2, direction=-1, seed=1101
    )

    # S12a/b/c same geometry as S01 (id dtype varies at load time, not here).
    for suffix in ("a", "b", "c"):
        scenes[f"S12{suffix}"] = [dict(r, scene_id=f"S12{suffix}") for r in scenes["S01"]]

    return scenes


def _reflect_rows(rows, new_scene):
    out = []
    for r in rows:
        r2 = dict(r, scene_id=new_scene)
        r2["x"] = _snap(105.0 - r["x"]) if np.isfinite(r["x"]) else r["x"]
        r2["y"] = _snap(68.0 - r["y"]) if np.isfinite(r["y"]) else r["y"]
        r2["vx"] = _snap(-r["vx"]) if np.isfinite(r["vx"]) else r["vx"]
        r2["vy"] = _snap(-r["vy"]) if np.isfinite(r["vy"]) else r["vy"]
        r2["dir"] = -r["dir"]
        out.append(r2)
    return out


def _xc_passes():
    """12 xC passes on S01/S02/S10 frames (recorded with the pass's own frame)."""
    rows = [
        dict(scene="X01", g=1, p=1, f=0, pid=100, tid=1, sx=-5.0, sy=0.0, ex=20.0, ey=10.0),
        dict(scene="X01", g=1, p=1, f=0, pid=101, tid=1, sx=0.0, sy=-10.0, ex=30.0, ey=-5.0),
        dict(scene="X01", g=1, p=1, f=1, pid=102, tid=1, sx=5.0, sy=5.0, ex=25.0, ey=0.0),
        dict(scene="X01", g=1, p=1, f=1, pid=103, tid=1, sx=-10.0, sy=10.0, ex=15.0, ey=20.0),
        dict(scene="X01", g=1, p=1, f=2, pid=104, tid=1, sx=0.0, sy=0.0, ex=40.0, ey=0.0),
        dict(scene="X01", g=1, p=1, f=2, pid=105, tid=1, sx=10.0, sy=-15.0, ex=35.0, ey=25.0),
    ]
    # scene tag refers to which scene's frames supply tracking; keep on S01 for all 6 here.
    return rows


# --------------------------------------------------------------------------------------------------
# Library invocation
# --------------------------------------------------------------------------------------------------


def _lib_frames(df):
    """A library-input copy: centre coords, string object ids, ball player_id -> 'ball', ball team None."""
    out = df.copy()
    out["x"] = out["x"].astype("float64") - _X_OFF
    out["y"] = out["y"].astype("float64") - _Y_OFF
    out["vx"] = out["vx"].astype("float64")
    out["vy"] = out["vy"].astype("float64")
    is_ball = out["is_ball"].to_numpy(dtype=bool)
    pid = out["player_id"].astype("object")
    pid[is_ball] = "ball"
    pid[~is_ball] = ["p" + str(int(v)) for v in out.loc[~is_ball, "player_id"]]
    out["player_id"] = pid
    tid = out["team_id"].astype("object")
    tid[is_ball] = None
    tid[~is_ball] = ["t" + str(int(v)) for v in out.loc[~is_ball, "team_id"]]
    out["team_id"] = tid
    out["team_in_possession"] = ["t" + str(int(v)) if pd.notna(v) else None for v in out["team_in_possession"]]
    return out


_COMMON = dict(
    frame_col="frame_id",
    player_col="player_id",
    team_col="team_id",
    x_col="x",
    y_col="y",
    vx_col="vx",
    vy_col="vy",
    team_in_possession_col="team_in_possession",
    ball_player_id="ball",
    period_col="period_id",
    attacking_direction_col="dir",
    infer_attacking_direction=False,
    use_progress_bar=False,
)


def _run_team_and_player(asp, df):
    """Return (team_rows, player_rows) lists for one (game,period) slice already prepared."""
    lib = _lib_frames(df)
    team = asp.get_dangerous_accessible_space(lib.copy(), **_COMMON)
    ind = asp.get_individual_dangerous_accessible_space(lib.copy(), **_COMMON)
    return team, ind


def _emit_team_player(scene, df, team, ind, team_rows, player_rows):
    base = df.reset_index(drop=True)
    as_t = np.asarray(team.acc_space, dtype=float)
    das_t = np.asarray(team.das, dtype=float)
    as_p = np.asarray(ind.player_acc_space, dtype=float)
    das_p = np.asarray(ind.player_das, dtype=float)
    # team.acc_space aligns to the possession-filtered rows; reindex onto base by position of kept rows.
    kept = base["team_in_possession"].notna().to_numpy()
    ti = 0
    per_frame = {}
    for i in range(len(base)):
        if not kept[i]:
            continue
        r = base.iloc[i]
        key = (int(r["game_id"]), int(r["period_id"]), int(r["frame_id"]))
        per_frame.setdefault(key, (as_t[ti], das_t[ti]))
        if not bool(r["is_ball"]):
            player_rows.append(
                dict(
                    scene_id=scene,
                    game_id=key[0],
                    period_id=key[1],
                    frame_id=key[2],
                    player_id=int(r["player_id"]),
                    AS=as_p[ti],
                    DAS=das_p[ti],
                )
            )
        ti += 1
    for (g, p, f), (a, d) in per_frame.items():
        team_rows.append(dict(scene_id=scene, game_id=g, period_id=p, frame_id=f, AS=a, DAS=d))


def _generate(out_dir: Path):
    import accessible_space as asp  # pyright: ignore[reportMissingImports]  # das-reference dev extra only, absent in CI

    ver = importlib.metadata.version("accessible_space")
    if ver != "2.0.15":
        raise SystemExit(f"golden is pinned to accessible-space 2.0.15, found {ver}")

    warnings.filterwarnings("ignore")

    scenes = _normal_scenes()
    scenes["M01"] = _reflect_rows(scenes["S01"], "M01")
    scenes["M02"] = _reflect_rows(scenes["S05"], "M02")
    scenes["M03"] = _reflect_rows(scenes["S10"], "M03")

    all_frame_rows: list[dict] = []
    for rows in scenes.values():
        all_frame_rows.extend(rows)

    team_rows: list[dict] = []
    player_rows: list[dict] = []
    for scene, rows in scenes.items():
        df = pd.DataFrame(rows)
        for (_g, _p), sl in df.groupby(["game_id", "period_id"], sort=True):
            team, ind = _run_team_and_player(asp, sl)
            _emit_team_player(scene, sl, team, ind, team_rows, player_rows)

    # xC passes (tracking from S01 frames).
    s01 = pd.DataFrame(scenes["S01"])
    xc_specs = _xc_passes()
    pass_rows = []
    xc_rows = []
    lib_track = _lib_frames(s01)
    passes_df = pd.DataFrame(
        [
            dict(
                frame_id=s["f"],
                player_id="p" + str(s["pid"]),
                team_id="t" + str(s["tid"]),
                x=s["sx"],
                y=s["sy"],
                x_target=s["ex"],
                y_target=s["ey"],
            )
            for s in xc_specs
        ]
    )
    ret = asp.get_expected_pass_completion(
        passes_df.copy(),
        lib_track.copy(),
        event_frame_col="frame_id",
        event_player_col="player_id",
        event_team_col="team_id",
        event_start_x_col="x",
        event_start_y_col="y",
        event_end_x_col="x_target",
        event_end_y_col="y_target",
        tracking_frame_col="frame_id",
        tracking_player_col="player_id",
        tracking_team_col="team_id",
        tracking_x_col="x",
        tracking_y_col="y",
        tracking_vx_col="vx",
        tracking_vy_col="vy",
        tracking_period_col="period_id",
        ball_tracking_player_id="ball",
        infer_attacking_direction=True,
        use_progress_bar=False,
    )
    xc_vals = np.asarray(ret.xc, dtype=float)
    for i, s in enumerate(xc_specs):
        pass_rows.append(
            dict(
                scene_id="X01",
                pass_index=i,
                game_id=s["g"],
                period_id=s["p"],
                frame_id=s["f"],
                player_id=s["pid"],
                team_id=s["tid"],
                start_x=_snap(float(s["sx"]) + _X_OFF),
                start_y=_snap(float(s["sy"]) + _Y_OFF),
                end_x=_snap(float(s["ex"]) + _X_OFF),
                end_y=_snap(float(s["ey"]) + _Y_OFF),
            )
        )
        xc_rows.append(dict(scene_id="X01", pass_index=i, xC=float(xc_vals[i])))

    errors = _divergence_scenes(asp, all_frame_rows, team_rows, player_rows, pass_rows, xc_rows)

    _write(out_dir, all_frame_rows, pass_rows, team_rows, player_rows, xc_rows, errors, ver, asp)


def _divergence_scenes(asp, all_frame_rows, team_rows, player_rows, pass_rows, xc_rows):
    """Run each defect scene the defect-exhibiting way; record a value or an exception."""
    errors: dict[str, dict] = {}

    def one_gp(
        scene, poss=1, direction: float = +1, n1=11, n2=11, ball_xy=(52.5, 34.0), seed=9001, mutate=None, frames=(0, 1)
    ):
        rows = []
        for f in frames:
            rows += _frame(
                scene,
                1,
                1,
                f,
                poss=poss,
                direction=direction,
                seed=seed + f,
                n1=n1,
                n2=n2,
                ball_xy=ball_xy,
                mutate=mutate,
            )
        return rows

    def record(scene, rows, *, single_call_no_period=False, is_pass=None):
        all_frame_rows.extend(rows)
        df = pd.DataFrame(rows)
        try:
            if single_call_no_period:
                lib = _lib_frames(df)
                common: dict = dict(_COMMON)
                common["period_col"] = None
                team = asp.get_dangerous_accessible_space(lib.copy(), **common)
                ind = asp.get_individual_dangerous_accessible_space(lib.copy(), **common)
                _emit_defect_team_player(scene, df, team, ind, team_rows, player_rows)
            else:
                for (_g, _p), sl in df.groupby(["game_id", "period_id"], sort=True):
                    team, ind = _run_team_and_player(asp, sl)
                    _emit_defect_team_player(scene, sl, team, ind, team_rows, player_rows)
        except Exception as exc:
            errors[scene] = {"type": type(exc).__name__, "message": _canonical_msg(str(exc))[:300]}

    # V-KEY: same frame_id in two periods, ONE call without period separation.
    vkey = _frame("V-KEY", 1, 1, 0, poss=1, direction=+1, seed=9100) + _frame(
        "V-KEY", 1, 2, 0, poss=1, direction=+1, seed=9101
    )
    record("V-KEY", vkey, single_call_no_period=True)

    # V-BALLNAN: NaN ball position.
    def ballnan(rows):
        for r in rows:
            if r["is_ball"]:
                r["x"] = float("nan")
                r["y"] = float("nan")
        return rows

    record("V-BALLNAN", one_gp("V-BALLNAN", seed=9200, mutate=ballnan))

    # V-OFF: one defender only (offside line from an arbitrary masked value).
    record("V-OFF", one_gp("V-OFF", n2=1, seed=9300))

    # V-PASSER: rows reversed, carrier (100) beyond the line -> row-order passer misassignment.
    def passer_mut(rows):
        for r in rows:
            if r["team_id"] == 2 and not r["is_ball"]:
                r["x"] = _snap(30.0)
            if r["player_id"] == 100:
                r["x"] = _snap(92.0)
            if r["is_ball"]:
                r["x"] = _snap(50.0)
        return list(reversed(rows))

    record("V-PASSER", one_gp("V-PASSER", seed=9400, mutate=passer_mut))

    # V-POSSABSENT: possession team absent from the frame (poss=9, only teams 1/2 present).
    def possabsent(rows):
        for r in rows:
            r["team_in_possession"] = 9
        return rows

    record("V-POSSABSENT", one_gp("V-POSSABSENT", seed=9500, mutate=possabsent))

    # V-DUP: a duplicate (frame, player) row.
    def dup(rows):
        return [*rows, dict(rows[0])]

    record("V-DUP", one_gp("V-DUP", seed=9600, mutate=dup, frames=(0,)))

    # V-MULTIBALL: two ball rows in a frame.
    def multiball(rows):
        extra = dict(next(r for r in rows if r["is_ball"]))
        extra["x"] = _snap(40.0)
        return [*rows, extra]

    record("V-MULTIBALL", one_gp("V-MULTIBALL", seed=9700, mutate=multiball, frames=(0,)))

    # V-DIRVAL: a non +-1 direction value (0.5) scales coordinates.
    dirval = one_gp("V-DIRVAL", direction=0.5, seed=9800)
    record("V-DIRVAL", dirval)

    # V-POSSVAR: possession varies within a frame.
    def possvar(rows):
        half = len(rows) // 2
        for r in rows[:half]:
            r["team_in_possession"] = 1
        for r in rows[half:]:
            if not r["is_ball"]:
                r["team_in_possession"] = 2
        return rows

    record("V-POSSVAR", one_gp("V-POSSVAR", seed=9900, mutate=possvar, frames=(0,)))

    # V-XC-FRAME: a pass whose frame is absent from tracking.
    s01 = pd.DataFrame([r for r in all_frame_rows if r["scene_id"] == "S01"])
    lib_track = _lib_frames(s01)
    try:
        bad = pd.DataFrame(
            [dict(frame_id=999, player_id="p100", team_id="t1", x=0.0, y=0.0, x_target=10.0, y_target=0.0)]
        )
        asp.get_expected_pass_completion(
            bad,
            lib_track.copy(),
            event_frame_col="frame_id",
            event_player_col="player_id",
            event_team_col="team_id",
            event_start_x_col="x",
            event_start_y_col="y",
            event_end_x_col="x_target",
            event_end_y_col="y_target",
            tracking_frame_col="frame_id",
            tracking_player_col="player_id",
            tracking_team_col="team_id",
            tracking_x_col="x",
            tracking_y_col="y",
            tracking_vx_col="vx",
            tracking_vy_col="vy",
            tracking_period_col="period_id",
            ball_tracking_player_id="ball",
            infer_attacking_direction=True,
            use_progress_bar=False,
        )
    except Exception as exc:
        errors["V-XC-FRAME"] = {"type": type(exc).__name__, "message": _canonical_msg(str(exc))[:300]}

    # V-XC-TEAM: a pass whose team is absent from tracking.
    try:
        bad = pd.DataFrame([dict(frame_id=0, player_id="pX", team_id="t9", x=0.0, y=0.0, x_target=10.0, y_target=0.0)])
        asp.get_expected_pass_completion(
            bad,
            lib_track.copy(),
            event_frame_col="frame_id",
            event_player_col="player_id",
            event_team_col="team_id",
            event_start_x_col="x",
            event_start_y_col="y",
            event_end_x_col="x_target",
            event_end_y_col="y_target",
            tracking_frame_col="frame_id",
            tracking_player_col="player_id",
            tracking_team_col="team_id",
            tracking_x_col="x",
            tracking_y_col="y",
            tracking_vx_col="vx",
            tracking_vy_col="vy",
            tracking_period_col="period_id",
            ball_tracking_player_id="ball",
            infer_attacking_direction=True,
            use_progress_bar=False,
        )
    except Exception as exc:
        errors["V-XC-TEAM"] = {"type": type(exc).__name__, "message": _canonical_msg(str(exc))[:300]}

    return errors


def _emit_defect_team_player(scene, df, team, ind, team_rows, player_rows):
    """Like _emit_team_player but tolerant of the library dropping/reordering rows in defect cases."""
    base = df.reset_index(drop=True)
    kept = base["team_in_possession"].notna().to_numpy()
    as_t = np.asarray(team.acc_space, dtype=float)
    das_t = np.asarray(team.das, dtype=float)
    n = min(int(kept.sum()), len(as_t))
    ti = 0
    per_frame = {}
    for i in range(len(base)):
        if not kept[i] or ti >= n:
            continue
        r = base.iloc[i]
        key = (int(r["game_id"]), int(r["period_id"]), int(r["frame_id"]))
        per_frame.setdefault(key, (as_t[ti], das_t[ti]))
        ti += 1
    for (g, p, f), (a, d) in per_frame.items():
        team_rows.append(dict(scene_id=scene, game_id=g, period_id=p, frame_id=f, AS=a, DAS=d))


# --------------------------------------------------------------------------------------------------
# Deterministic serialisation
# --------------------------------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict], cols: list[str], sort_keys: list[str]):
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    if len(df):
        df = df.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    lines = [",".join(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, bool):
                cells.append("True" if v else "False")
            elif v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append("")
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):.17g}")
            else:
                cells.append(str(v))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _write(out_dir, frame_rows, pass_rows, team_rows, player_rows, xc_rows, errors, ver, asp):
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        out_dir / "scenes_frames.csv",
        frame_rows,
        _FRAME_COLS,
        ["scene_id", "game_id", "period_id", "frame_id", "is_ball", "player_id"],
    )
    _write_csv(out_dir / "scenes_passes.csv", pass_rows, _PASS_COLS, ["scene_id", "pass_index"])
    _write_csv(
        out_dir / "reference_das_team.csv",
        team_rows,
        ["scene_id", "game_id", "period_id", "frame_id", "AS", "DAS"],
        ["scene_id", "game_id", "period_id", "frame_id"],
    )
    _write_csv(
        out_dir / "reference_das_player.csv",
        player_rows,
        ["scene_id", "game_id", "period_id", "frame_id", "player_id", "AS", "DAS"],
        ["scene_id", "game_id", "period_id", "frame_id", "player_id"],
    )
    _write_csv(out_dir / "reference_xc.csv", xc_rows, ["scene_id", "pass_index", "xC"], ["scene_id", "pass_index"])
    (out_dir / "reference_errors.json").write_text(
        json.dumps(errors, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    asp_dir = Path(asp.__file__).resolve().parent
    src_sha = {
        name: hashlib.sha256((asp_dir / name).read_bytes()).hexdigest()
        for name in ("core.py", "interface.py", "utility.py", "motion_models.py")
    }
    import accessible_space.core as _core  # pyright: ignore[reportMissingImports]  # das-reference dev-only
    import accessible_space.interface as _iface  # pyright: ignore[reportMissingImports]  # das-reference dev-only

    defaults = {f"core.{k}": getattr(_core, k) for k in dir(_core) if k.startswith("_DEFAULT")}
    defaults.update({f"interface.{k}": getattr(_iface, k) for k in dir(_iface) if k.startswith("_DEFAULT")})
    meta = {
        "accessible_space_version": ver,
        "accessible_space_source_sha256": src_sha,
        "accessible_space_defaults": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in sorted(defaults.items())
        },
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": __import__("scipy").__version__,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "column_dtypes": {
            "player_id": "Int64",
            "team_id": "Int64",
            "team_in_possession": "Int64",
            "x": "float32",
            "y": "float32",
            "vx": "float32",
            "vy": "float32",
            "is_ball": "bool",
            "dir": "float64",
        },
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    names = [
        "scenes_frames.csv",
        "scenes_passes.csv",
        "reference_das_team.csv",
        "reference_das_player.csv",
        "reference_xc.csv",
        "reference_errors.json",
        "metadata.json",
    ]
    sums = [f"{hashlib.sha256((out_dir / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (out_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print(
        f"wrote golden fixture to {out_dir} ({len(frame_rows)} frame rows, {len(team_rows)} team, "
        f"{len(player_rows)} player, {len(xc_rows)} xc, {len(errors)} errors)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_HERE))
    args = ap.parse_args()
    _generate(Path(args.out))


if __name__ == "__main__":
    main()
