"""Paired Leg A / Leg B fixture for the SB360 audit.

Leg A is built by CALLING ``snapshot_to_tracking_frames`` -- never hand-assembled -- so the
fixture cannot drift from the producer and the audit exercises the path real SB360 data hits.
Leg B is linked by the real ``link_actions_to_frames`` for the same reason.

``FIXTURE_VERSION`` is surfaced in every observation-lock failure message: the lock pins the
fixture as well as the library, so "the fixture changed" and "the library regressed" must be
distinguishable at the point of failure.

**Orientation is deliberately uniform.** Both legs label every row ``team_attacking_direction
= "ltr"``. Leg A must, because snapshot frames are already in SPADL action-LTR
(``_action_orientation.py:56-58``). Leg B matches it so the ADR-028 re-projection is a no-op
on BOTH legs -- a home-attacks-right Leg B would re-project away-team actions and break the
per-linked-frame position equality the whole comparison rests on. The consequence, stated
rather than hidden: this fixture does NOT exercise orientation handling. That is
``tests/tracking/test_mirror_registry.py``'s job, not this audit's.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

import functools
import inspect

import numpy as np
import pandas as pd

from silly_kicks.tracking import link_actions_to_frames, snapshot_to_tracking_frames

FIXTURE_VERSION = "sb360-fixture-1"

HOME_TEAM_ID = 1
AWAY_TEAM_ID = 2

_FRAME_HZ = 10.0
_GAME_ID = 7
_PERIOD = 1

#: (action_id, type_name, team_id, player_id, start_x, start_y, end_x, end_y, time_seconds)
_ACTIONS: tuple[tuple, ...] = (
    (0, "pass", HOME_TEAM_ID, 10, 52.5, 34.0, 70.0, 40.0, 300.0),
    (1, "cross", HOME_TEAM_ID, 11, 88.0, 8.0, 98.0, 34.0, 320.0),
    (2, "shot", HOME_TEAM_ID, 12, 95.0, 34.0, 105.0, 34.0, 340.0),
    (3, "goalkick", AWAY_TEAM_ID, 20, 5.5, 34.0, 45.0, 20.0, 360.0),
    (4, "dribble", HOME_TEAM_ID, 13, 60.0, 50.0, 68.0, 52.0, 380.0),
    (5, "throw_in", AWAY_TEAM_ID, 21, 40.0, 68.0, 48.0, 55.0, 400.0),
)

#: ALLOWLIST of parameter names that are genuinely temporal frame windows.
#:
#: A denylist was tried and is wrong on both sides. Matching any name containing "seconds"
#: admits ``add_gk_influence.tau_seconds`` and ``add_player_influence.tau_seconds``, which are
#: influence-DECAY constants rather than frame windows -- so the scan measures the wrong set
#: and returns the right answer by luck. Worse, a denylist is open at the top: a future
#: ``timeout_seconds=600.0`` anywhere in ``tracking.__all__`` would silently inflate Leg B into
#: the millions of rows. Fail-safe defaults, the rule the repo applies elsewhere.
_WINDOW_PARAM_NAMES = frozenset({"pre_seconds", "post_seconds", "window_seconds"})

#: Upper bound on a single discovered window. Leg B is 10 Hz x 6 actions x 23 rows, so its
#: row count scales as ~5500 x this value; 10 s is ~55k rows, 600 s would be ~3.3M.
_MAX_PLAUSIBLE_WINDOW_S = 10.0


def discovered_windows() -> dict[str, float]:
    """Every temporal-window default across the enumerated aggregators, by ``func.param``."""
    import silly_kicks.tracking as T

    found: dict[str, float] = {}
    for fn_name in (n for n in T.__all__ if n.startswith("add_")):
        fn = getattr(T, fn_name)
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        for pname, p in params.items():
            if pname not in _WINDOW_PARAM_NAMES:
                continue
            if isinstance(p.default, (int, float)) and not isinstance(p.default, bool):
                found[f"{fn_name}.{pname}"] = float(p.default)
    return found


def required_neighbourhood_seconds() -> float:
    """Longest window among ALL enumerated features, read from the library, never hardcoded.

    Scans every ``add_*`` rather than one function, because the claim is "longest among
    enumerated features" and reading a single signature does not establish it. Both bounds are
    asserted: an empty result means the scan silently returned a fallback, and an implausible
    maximum means it admitted something that is not a frame window.
    """
    windows = discovered_windows()
    assert windows, (
        f"no temporal-window parameter discovered across tracking.__all__ from allowlist "
        f"{sorted(_WINDOW_PARAM_NAMES)}. The fixture would fall back to a hardcoded value "
        f"while claiming to read the library. A new window parameter must be added here."
    )
    longest = max(windows.values())
    assert longest <= _MAX_PLAUSIBLE_WINDOW_S, (
        f"longest discovered window is {longest}s (> {_MAX_PLAUSIBLE_WINDOW_S}s): "
        f"{ {k: v for k, v in windows.items() if v == longest} }. Leg B would balloon to "
        f"roughly {int(5500 * longest):,} rows. Either the allowlist admitted a non-window "
        f"parameter, or the fixture needs a deliberate redesign."
    )
    # 2x headroom so a boundary-inclusive implementation still has frames to consume.
    return 2.0 * longest


def _player_layout(roster: str) -> list[dict]:
    """22 players in a plausible shape, one keeper per side.

    Returns RECORDS, not a DataFrame, so call sites ``enumerate`` a list rather than walking
    ``iterrows()`` -- whose index is ``Hashable`` and needs a cast at every use.
    """
    rows: list[dict] = []
    for i in range(11):
        rows.append(
            {
                "player_id": 10 + i,
                "team_id": HOME_TEAM_ID,
                "is_goalkeeper": i == 0,
                "base_x": 5.0 if i == 0 else 30.0 + (i % 4) * 18.0,
                "base_y": 34.0 if i == 0 else 8.0 + (i % 5) * 13.0,
            }
        )
    for i in range(11):
        rows.append(
            {
                "player_id": 20 + i,
                "team_id": AWAY_TEAM_ID,
                "is_goalkeeper": i == 0,
                "base_x": 100.0 if i == 0 else 60.0 + (i % 4) * 11.0,
                "base_y": 34.0 if i == 0 else 10.0 + (i % 5) * 12.0,
            }
        )
    if roster == "gk_absent":
        rows = [r for r in rows if not r["is_goalkeeper"]]
    elif roster == "defender_absent":
        # Drop one outfield away player positioned far from the action: the extreme-member
        # case, which is what the applicability probe's probe 1 leans on.
        rows = [r for r in rows if r["player_id"] != 24]
    elif roster == "gk_one_end":
        # ONE keeper visible, the other off-frame. The DEFENDING keeper is in-frame on 92.2% of
        # shots (`docs/research/sb360_coverage/coverage.md`), so a freeze-frame WITH a keeper is
        # the common shape and `gk_absent` alone leaves it unexercised. Deliberately NOT claimed
        # here: that the far keeper is usually absent. That report's `acting GK` cell for `shot`
        # is "—", meaning definitionally not applicable (the keeper is not the actor on a shot),
        # NOT a measured low rate -- it says nothing about the far keeper either way.
        #
        # This roster exists to break `gk_absent`'s DEGENERACY rather than to replace it
        # (`gk_absent` is a real visibility axis and the only case exercising the both-absent
        # refusal path).
        #
        # Keeping the HOME keeper (base_x 5.0) makes team 1 RESOLVE to x=0. Team 2 falls to the
        # outfield rung, whose ten members sit at `60 + (i % 4) * 11` -> 71/82/93/60/... for a
        # mean of 76.5, above the 52.5 midline, so it GUESSES x=105. The two ends DIFFER, the map
        # is non-degenerate, `attacked_goal` resolves, and the five `add_cover_shadows` columns
        # become exercisable again (ADR-055 made that aggregator keeper-dependent).
        rows = [r for r in rows if not (r["is_goalkeeper"] and r["team_id"] == AWAY_TEAM_ID)]
    return rows


@functools.cache
def _type_ids() -> dict[str, int]:
    """SPADL ``type_name`` -> ``type_id``, read from the library's own config table."""
    import silly_kicks.spadl as spadl

    df = spadl.actiontypes_df()
    return {str(n): int(i) for i, n in zip(df["type_id"], df["type_name"], strict=True)}


def _actions_frame() -> pd.DataFrame:
    """Actions carrying the FULL canonical SPADL schema, with correct dtypes.

    An earlier draft emitted only ``type_name`` plus coordinates. Four aggregators raised
    ``KeyError: 'type_id'`` on it -- and a fixture whose actions are not real SPADL would have
    produced verdicts about a shape the library never sees. ``type_name`` is kept ALONGSIDE the
    schema (it is not a SPADL column) because the audit's own reporting reads it.
    """
    tid = _type_ids()
    rows = []
    for aid, tname, team, pid, sx, sy, ex, ey, t in _ACTIONS:
        rows.append(
            {
                "game_id": _GAME_ID,
                "original_event_id": f"evt_{aid}",
                "action_id": aid,
                "period_id": _PERIOD,
                "time_seconds": t,
                "team_id": team,
                "player_id": pid,
                "start_x": sx,
                "start_y": sy,
                "end_x": ex,
                "end_y": ey,
                "type_id": tid[tname],
                # Every action succeeds. A fixture mixing outcomes would make an
                # outcome-reading feature differ between legs for a reason unrelated to
                # either axis.
                "result_id": 1,
                "bodypart_id": 0,
                "shot_blocked": pd.NA,
                "cross_blocked": pd.NA,
                "type_name": tname,
            }
        )
    out = pd.DataFrame(rows)
    from silly_kicks.spadl.schema import SPADL_COLUMNS

    for col, dtype in SPADL_COLUMNS.items():
        out[col] = out[col].astype(dtype)  # type: ignore[arg-type]
    return out


def _omega(player_index: int) -> float:
    return 0.35 + 0.05 * player_index


def _offset(action_id: int, player_index: int, t: float) -> tuple[float, float]:
    """Non-degenerate trajectory: speed AND heading both vary with time.

    A constant-velocity path zeroes every acceleration-dependent quantity in BOTH legs, which
    the comparison reads as ``identical`` and the audit records as ``works``.
    """
    w = _omega(player_index)
    return (
        3.0 * np.sin(w * t + action_id),
        2.0 * np.sin(0.5 * w * t + 0.7 * player_index),
    )


def _velocity(action_id: int, player_index: int, t: float) -> tuple[float, float]:
    """Analytic d/dt of ``_offset``, so velocity never contradicts position (ADR-045 D1)."""
    w = _omega(player_index)
    return (
        3.0 * w * np.cos(w * t + action_id),
        2.0 * 0.5 * w * np.cos(0.5 * w * t + 0.7 * player_index),
    )


#: How long the ball takes to travel from an action's start to its end, in Leg B.
_BALL_FLIGHT_S = 1.0


def _ball_state(t: float, t0: float, sx: float, sy: float, ex: float, ey: float):
    """Ball position/velocity at ``t`` for an action kicked at ``t0``.

    A STATIC ball was tried and measured to break two aggregators for the same reason: with no
    kick to detect, ``add_elastic_sync`` returned NaN on every Leg B row (``leg_b_declined``)
    and ``add_shot_goalmouth`` had no trajectory to fit (``no_signal``). Both would have been
    recorded as fixture inadequacies rather than library properties -- honest, but a wasted
    verdict.

    The ball rests at the action's start until ``t0``, then travels to the end over
    ``_BALL_FLIGHT_S`` with a parabolic height. **At ``t == t0`` it is exactly at
    ``(sx, sy)``** -- the anchor frame is what Leg A's producer places there, so the
    per-linked-frame position invariant is preserved by construction.
    """
    if t <= t0:
        return float(sx), float(sy), 0.0, 0.0, 0.0
    frac = min((t - t0) / _BALL_FLIGHT_S, 1.0)
    x = float(sx + (ex - sx) * frac)
    y = float(sy + (ey - sy) * frac)
    # Parabolic arc, peaking mid-flight; zero once the ball has arrived.
    z = float(4.0 * 2.5 * frac * (1.0 - frac)) if frac < 1.0 else 0.0
    if frac >= 1.0:
        return x, y, z, 0.0, 0.0
    return x, y, z, float((ex - sx) / _BALL_FLIGHT_S), float((ey - sy) / _BALL_FLIGHT_S)


def _cast_ids(df: pd.DataFrame, id_dtype: str) -> pd.DataFrame:
    out = df.copy()
    for col in ("team_id", "player_id"):
        if col in out.columns:
            out[col] = out[col].astype(id_dtype)  # type: ignore[arg-type]
    return out


@functools.cache
def frame_id_dtype(id_dtype: str) -> str:
    """What the REAL producer makes of a frame's id columns, for a given action id dtype.

    DERIVED by asking ``snapshot_to_tracking_frames``, never hardcoded -- because the answer
    is **pandas-version-dependent** and a hardcoded table passes on one interpreter and fails
    on the other. Measured:

    ==========  ================  ================
    input       pandas 2.3.3      pandas 3.0.3
    ==========  ================  ================
    ``int64``   ``float64``       ``float64``
    ``Int64``   ``Int64``         ``Float64``
    ``object``  ``object``        ``object``
    ==========  ================  ================

    A frame set carries a ball row whose ids are NA, so the producer's concat must widen; what
    it widens a NULLABLE integer to changed in pandas 3 (the ``FutureWarning`` that call emits
    about concat with all-NA entries, materialised). An earlier draft hardcoded the pandas-2
    row and failed the full suite on ``.venv312`` with ``assert 'Float64' == 'Int64'``.

    The invariant that matters is that BOTH legs land on the same dtype, so both ask here.
    """
    probe_actions = pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [_GAME_ID],
            "period_id": [_PERIOD],
            "time_seconds": [0.0],
            "start_x": [52.5],
            "start_y": [34.0],
        }
    )
    probe_snapshots = _cast_ids(
        pd.DataFrame(
            {
                "action_id": [0],
                "team_id": [HOME_TEAM_ID],
                "player_id": [10],
                "is_goalkeeper": [False],
                "x": [50.0],
                "y": [30.0],
            }
        ),
        id_dtype,
    )
    probe_frames, _ = snapshot_to_tracking_frames(probe_snapshots, probe_actions)
    return str(probe_frames["player_id"].dtype)


def _cast_frame_ids(df: pd.DataFrame, id_dtype: str) -> pd.DataFrame:
    return _cast_ids(df, frame_id_dtype(id_dtype))


def build_leg_a(*, roster: str = "full", id_dtype: str = "int64"):
    """Freeze-frame leg: one synthetic frame per action, via the real producer."""
    # Cast BEFORE the producer sees it, not after. Casting only on the way out means the
    # producer always runs on one dtype combination and the parameterization never exercises
    # the ADR-019 `ids_match` path it exists to probe.
    actions = _cast_ids(_actions_frame(), id_dtype)
    layout = _player_layout(roster)

    snap_rows = []
    for aid, _, _, _, _, _, _, _, t in _ACTIONS:
        for idx, p in enumerate(layout):
            dx, dy = _offset(aid, idx, t)
            snap_rows.append(
                {
                    "action_id": aid,
                    "team_id": p["team_id"],
                    "player_id": p["player_id"],
                    "is_goalkeeper": bool(p["is_goalkeeper"]),
                    "x": float(p["base_x"] + dx),
                    "y": float(p["base_y"] + dy),
                }
            )
    snapshots = _cast_ids(pd.DataFrame(snap_rows), id_dtype)
    frames, links = snapshot_to_tracking_frames(snapshots, actions)

    # `team_in_possession` is a CALLER-side enrichment (`derive_team_in_possession`), not part
    # of the 20-column schema the producer emits. Both legs receive it identically, because
    # without it the comparison is unfair rather than informative: Leg A short-circuits on the
    # velocity marker BEFORE `add_das` checks for this column, so Leg A succeeded while Leg B
    # -- which has velocity and proceeds -- raised. That surfaced as `raises_b`, correctly
    # flagged as a fixture defect rather than recorded as a library property.
    possession = {aid: team for aid, _, team, _, _, _, _, _, _ in _ACTIONS}
    frames = frames.copy()
    frames["team_in_possession"] = frames["frame_id"].map(possession)
    return actions, frames, links


def build_leg_b(*, roster: str = "full", id_dtype: str = "int64"):
    """Velocity-bearing leg: same positions at the linked frame, plus a real neighbourhood."""
    actions = _cast_ids(_actions_frame(), id_dtype)
    layout = _player_layout(roster)
    half = required_neighbourhood_seconds()
    step = 1.0 / _FRAME_HZ

    rows = []
    frame_id = 0
    n_steps = round(half / step)
    for aid, _, act_team, _, sx, sy, ex, ey, t0 in _ACTIONS:
        # Built as t0 + k*step over INTEGER k, so k=0 lands on t0 EXACTLY. `np.arange(t0-half,
        # ...)` accumulates float drift and put the anchor ~1e-12 off t0, which made the
        # per-linked-frame position equality approximate. An exact invariant beats a tolerance
        # chosen to accommodate an avoidable error.
        times = t0 + np.arange(-n_steps, n_steps + 1) * step
        for t in times:
            for idx, p in enumerate(layout):
                # Position matches Leg A exactly at t == t0 by construction.
                dx, dy = _offset(aid, idx, float(t))
                vx, vy = _velocity(aid, idx, float(t))
                rows.append(
                    {
                        "game_id": _GAME_ID,
                        "period_id": _PERIOD,
                        "frame_id": frame_id,
                        "time_seconds": float(t),
                        "frame_rate": _FRAME_HZ,
                        "player_id": p["player_id"],
                        "team_id": p["team_id"],
                        "is_ball": False,
                        "is_goalkeeper": bool(p["is_goalkeeper"]),
                        "x": float(p["base_x"] + dx),
                        "y": float(p["base_y"] + dy),
                        "z": np.nan,
                        "speed": float(np.hypot(vx, vy)),
                        "vx": float(vx),
                        "vy": float(vy),
                        "speed_source": "derived",
                        "ball_state": "alive",
                        "team_in_possession": act_team,
                        "team_attacking_direction": "ltr",
                        "confidence": np.nan,
                        "visibility": np.nan,
                        "source_provider": "synthetic",
                        "is_goalkeeper_source": "native",
                    }
                )
            b_x, b_y, b_z, b_vx, b_vy = _ball_state(float(t), t0, sx, sy, ex, ey)
            rows.append(
                {
                    "game_id": _GAME_ID,
                    "period_id": _PERIOD,
                    "frame_id": frame_id,
                    "time_seconds": float(t),
                    "frame_rate": _FRAME_HZ,
                    "player_id": np.nan,
                    "team_id": np.nan,
                    "is_ball": True,
                    "is_goalkeeper": False,
                    "x": b_x,
                    "y": b_y,
                    "z": b_z,
                    "speed": float(np.hypot(b_vx, b_vy)),
                    "vx": b_vx,
                    "vy": b_vy,
                    "speed_source": "derived",
                    "ball_state": "alive",
                    "team_in_possession": act_team,
                    "team_attacking_direction": "ltr",
                    "confidence": np.nan,
                    "visibility": np.nan,
                    "source_provider": "synthetic",
                    "is_goalkeeper_source": "native",
                }
            )
            frame_id += 1

    frames = _cast_frame_ids(pd.DataFrame(rows), id_dtype)

    # Link with the REAL linker, for the same reason Leg A uses the real producer: a
    # hand-built five-column table drifts the moment the linkage contract changes, and the
    # LinkReport gives a free assertion that every action actually found a frame.
    #
    # `link_actions_to_frames` returns one row per action with a NaN frame_id when nothing
    # falls inside tolerance, so a length check would be trivially true -- the real assertion
    # is that every frame_id RESOLVED.
    links, report = link_actions_to_frames(actions, frames)
    unlinked = int(links["frame_id"].isna().sum())
    assert unlinked == 0, (
        f"{unlinked}/{len(actions)} actions did not link in Leg B (report={report}). An "
        f"unlinked action produces NaN geometry that would be misread as a library property."
    )
    return actions, frames, links


def build_leg_b_anchor_only(*, roster: str = "full", id_dtype: str = "int64"):
    """Leg B truncated to ONE frame per action -- the anchor -- keeping velocity.

    The DIAGNOSTIC leg, not a third axis. It exists because ``differs`` and ``all_nan`` each
    have two possible causes and the adjudication depends on which:

    * Leg A (1 frame, no velocity) vs **this** (1 frame, WITH velocity) isolates **velocity**.
    * This (1 frame) vs full Leg B (a 10 Hz neighbourhood) isolates **frame count**.

    Without the isolation, a feature that merely needs a temporal window is indistinguishable
    from one that fabricates a number out of absent kinematics -- and only the second is a
    finding. Writing ``silent_degrade`` off the un-isolated reading would be exactly the
    plausible-number-from-a-computation-that-did-not-happen defect this audit exists to find.
    """
    actions, frames, links = build_leg_b(roster=roster, id_dtype=id_dtype)
    anchor_ids = set(links["frame_id"].dropna().astype(int).tolist())
    kept = frames[frames["frame_id"].isin(anchor_ids)].reset_index(drop=True)
    return actions, kept, links


def visible_area_side_table(*, fraction: float = 1.0) -> pd.DataFrame:
    """Synthetic camera polygons, keyed by ``action_id``.

    A HARNESS-ONLY side table. The ``snapshots`` contract is NOT extended and
    ``snapshot_to_tracking_frames`` is NOT modified -- stating the seam here closes the route
    by which scope creeps into a public contract.
    """
    rows = []
    for aid, _, _, _, sx, _, _, _, _ in _ACTIONS:
        half_len = 52.5 * fraction
        rows.append(
            {
                "action_id": aid,
                "polygon": [
                    (max(0.0, sx - half_len), 0.0),
                    (min(105.0, sx + half_len), 0.0),
                    (min(105.0, sx + half_len), 68.0),
                    (max(0.0, sx - half_len), 68.0),
                ],
            }
        )
    return pd.DataFrame(rows)
