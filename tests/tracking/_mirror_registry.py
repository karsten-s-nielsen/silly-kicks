"""Registry backing the ADR-028 mirror gates (spec 2026-07-29, section 6).

THREE gates, because one instrument cannot see the defect classes:

* **Gate A** -- physical mirror. Detects CONVENTION MIXING (an action-LTR value combined with a
  frame-LTR one). ``home_team_id`` is swapped, because after a physical mirror the team attacking
  +x really is the other one.
* **Gate B** -- ``home_team_id`` invariance on FIXED canonical frames. Detects IDENTITY-KEYED
  direction inference, which Gate A is structurally BLIND to: swapping ``home_team_id`` restores
  the very invariant identity-keying assumes, so an identity-keyed aggregator is invariant under
  Gate A whether it is safe or not.
* **Gate C** (ADR-055) -- ``goal_map`` DEPENDENCE on fixed canonical frames. Once an aggregator is
  re-keyed off ``home_team_id`` onto the map, Gate B's variable carries nothing and the gate goes
  vacuous (it SKIPS on ``role="unused"``). Gate C is the same question one variable further out:
  hold the frames fixed, swap the MAP, and require the invariant columns to MOVE. If nothing
  moves, the aggregator is not reading the map and the re-key was cosmetic.

  Gate C does NOT replace Gate B's correctness claim, only its DETECTION: ``get`` and
  ``attacked_goal`` both move when the map is swapped, so a moved column proves the map is
  consulted, not that the right accessor was chosen. That half is
  ``test_goal_map_consumers.py``.

Mirror classes (per emitted column):

``invariant``
    Action-LTR geometry; base and mirror identical. The default.
``mirrored_pitch_absolute``
    Deliberately pitch-absolute; must equal its OWN REFLECTION. Requires a ``reflections`` entry
    AND a reason -- without both, the class is a reason-free ``exempt`` under another name.
``exempt``
    Undefined or non-deterministic under mirror. REQUIRES a written reason.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
HOME, AWAY = 1, 2

MirrorClass = str  # "invariant" | "mirrored_pitch_absolute" | "exempt"
HomeRole = str  # "direction_only" | "attribution" | "unused"


@dataclass(frozen=True)
class MirrorEntry:
    """One aggregator's declaration. See the module docstring for the vocabularies."""

    name: str
    call: Callable  # (actions, frames, home_team_id) -> pd.DataFrame
    #: Gate C only: ``(actions, frames, goal_map) -> pd.DataFrame``. Non-``None`` IS the
    #: "this aggregator consumes a goal map" predicate -- deliberately NOT a separate boolean
    #: flag, which would be a second field free to disagree with the first.
    call_with_map: Callable | None
    columns: dict[str, MirrorClass]
    tolerance: float
    tolerance_basis: str
    home_team_id_role: HomeRole
    #: Columns that must be NON-NULL on away rows, or the comparison is vacuous.
    non_vacuity: tuple[str, ...]
    exempt_reasons: dict[str, str] = field(default_factory=dict)
    #: REQUIRED for every ``mirrored_pitch_absolute`` column: how to reflect the mirror-leg value
    #: before comparing. ``"x"`` -> FIELD_LENGTH - v, ``"y"`` -> FIELD_WIDTH - v, or a dict for a
    #: label swap (e.g. ``{"left": "right", "right": "left"}``).
    reflections: dict[str, str | dict] = field(default_factory=dict)
    #: Set for a KNOWN-BROKEN aggregator. Gate A xfails STRICTLY until the fix lands, so whoever
    #: fixes it is forced to delete the marker.
    known_defect: str | None = None
    #: Same, for Gate B (identity-keyed direction -- the D3 re-key targets).
    known_defect_gate_b: str | None = None
    #: Columns Gate B must NOT check, each with a reason. For a column whose dependence on
    #: ``home_team_id`` is genuine ATTRIBUTION rather than direction -- e.g. ``add_xcross_attempt``'s
    #: score is conditioned on ``score_differential``, whose SIGN is a match fact, not geometry.
    #:
    #: This exists because the vocabulary conflates two axes: a column can be mirror-invariant
    #: (Gate A) and legitimately identity-DEPENDENT (Gate B). Reusing the ``invariant`` mirror class
    #: for Gate B's surface silently assumed they were the same axis. A reason is REQUIRED, and
    #: when an entry exempts every column Gate B would check, the gate records the skip rather than
    #: tripping its own vacuity assert -- which is what an entry with exactly one attribution
    #: column did.
    gate_b_exempt: dict[str, str] = field(default_factory=dict)
    #: Gate C: columns that MUST move when the goal map is swapped. Named columns, not a bare
    #: ``moved > 0``, because "something moved" is satisfied by a partial re-key: ``add_gk_influence``
    #: reads the map down TWO independent paths (``_gk_influence_at_actions`` and
    #: ``_closing_time_per_series``), so a one-column result means the closing-time path was missed
    #: and would otherwise read as success. Required whenever ``call_with_map`` is set.
    gate_c_must_move: tuple[str, ...] = ()
    #: Optional per-entry scene builder, defaulting to :func:`canonical_scene`.
    #:
    #: The shared scene is the right default: every entry measured on ONE scene is what makes
    #: the recorded deltas comparable across the registry. But one scene cannot be
    #: simultaneously optimal for every aggregator -- packing needs a secured reception,
    #: off-ball runs need real displacement, and an entry that needs neither should not pay
    #: for them. Without this seam the ONLY way to give an entry a non-degenerate case is to
    #: mutate the scene ten other modules pin, which is precisely the coupling that made the
    #: degeneracy permanent. Set it only when the shared scene genuinely cannot serve the
    #: entry, and say why at the registration.
    scene: Callable[[], tuple[pd.DataFrame, pd.DataFrame]] | None = None


MIRROR_ENTRIES: dict[str, MirrorEntry] = {}


def _entry(
    name,
    call,
    columns,
    *,
    tol,
    basis,
    role,
    non_vacuity,
    exempt=None,
    reflections=None,
    defect=None,
    defect_b=None,
    gate_b_exempt=None,
    call_with_map=None,
    gate_c_must_move=(),
    scene=None,
) -> None:
    MIRROR_ENTRIES[name] = MirrorEntry(
        name=name,
        call=call,
        call_with_map=call_with_map,
        columns=columns,
        tolerance=tol,
        tolerance_basis=basis,
        home_team_id_role=role,
        non_vacuity=tuple(non_vacuity),
        exempt_reasons=exempt or {},
        reflections=reflections or {},
        known_defect=defect,
        known_defect_gate_b=defect_b,
        gate_b_exempt=gate_b_exempt or {},
        gate_c_must_move=tuple(gate_c_must_move),
        scene=scene,
    )


_BASE_FRAME = dict(
    game_id=1,
    period_id=1,
    frame_rate=25.0,
    z=0.0,
    speed=1.0,
    speed_source="native",
    ball_state="alive",
    confidence=None,
    visibility=True,
    source_provider="synthetic",
    is_goalkeeper_source="native",
)


@functools.cache
def canonical_scene() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(actions, frames)`` -- canonical converter shape: y-ASYMMETRIC and PHYSICALLY COHERENT.

    y-asymmetry is not decoration: an x-only reprojection is exact on a y-symmetric scene, and
    ADR-041 shipped precisely that incomplete repair -- only a y-asymmetric oracle caught it.

    Home attacks +x (``"ltr"``); away attacks -x (``"rtl"``). BOTH teams act, so the away rows --
    the only rows an ADR-028 defect touches -- are a real population rather than a single token
    action.

    COHERENCE (C0 / D7). Positions are DERIVED from ``base + v * (t - T_REF)``, so the declared
    ``vx``/``vy``/``speed`` columns and the observed inter-frame displacement are the same fact.
    The previous scene held every position CONSTANT while declaring ``vx=0.8, vy=-0.5,
    speed=1.0`` -- two contradictory answers to "how fast is this player moving", and which one
    an aggregator saw depended on whether it read the columns or differenced the frames. That is
    the defect class this registry exists to catch, sitting in the registry's own reference
    scene: a plausible number from a computation that had not happened.

    The old constancy carried a fence -- *"a positional drift would desynchronise the action
    anchors (which name a specific frame position) from the frame the linker actually picks, for
    no gain"*. The fence is respected and its problem is SOLVED rather than dodged: every action
    is anchored to ``_at(actor, t)``, the actor's position at that action's own timestamp, so the
    anchor tracks the trajectory by construction instead of being frozen to avoid it. The "for no
    gain" clause was written when the only benefit on offer was mirror invariance (Gate A); Gate C
    liveness and off-ball-run detection are gains it did not weigh.

    TEMPORAL AXIS. Actions are spread in time and the frame grid spans them, because several
    emitted quantities are only defined over an interval: ``packing_secured`` needs a resolvable
    reception plus a decisive follow-up (a same-team shot decides ``True`` immediately; an
    opponent possession boundary decides ``False``), and off-ball runs need displacement. With
    every action at one instant those columns are structurally ``<NA>`` no matter what the
    geometry says.
    """
    from silly_kicks.spadl import config as spadlconfig

    t_ref = 8.0
    # (player_id, team, is_gk, x@t_ref, y@t_ref, vx, vy) -- y spread ASYMMETRICALLY about y=34.
    # Velocities vary in magnitude and sign so off-ball run detection has a real distribution
    # rather than one repeated value; keepers drift slowly, outfielders carry.
    spec: list[tuple[int, int, bool, float, float, float, float]] = [
        (1, HOME, True, 5.0, 27.0, 0.30, 0.10),
        (50, AWAY, True, 100.0, 41.0, -0.25, 0.15),
    ]
    home_out = [
        (10, 28.0, 12.0, 1.60, 0.40),
        (11, 36.0, 21.0, 2.10, -0.60),
        (12, 44.0, 9.0, 0.90, 1.30),
        (13, 52.0, 30.0, 2.60, 0.20),
        (14, 60.0, 17.0, 1.10, -1.40),
        (15, 33.0, 44.0, 3.10, -0.70),
        (16, 47.0, 55.0, 0.50, 0.90),
        (17, 58.0, 38.0, 2.40, -1.10),
        (18, 25.0, 50.0, 1.80, 1.60),
        (19, 41.0, 62.0, 0.70, -2.20),
    ]
    away_out = [
        (60, 70.0, 14.0, -1.50, 0.80),
        (61, 63.0, 25.0, -0.60, -1.20),
        (62, 77.0, 8.0, -2.30, 0.50),
        (63, 55.0, 33.0, -1.90, -0.40),
        (64, 68.0, 19.0, -0.80, 1.70),
        (65, 74.0, 47.0, -2.70, -0.90),
        (66, 61.0, 58.0, -1.20, 0.30),
        (67, 80.0, 36.0, -0.40, -1.60),
        (68, 66.0, 52.0, -3.00, 1.10),
        (69, 50.0, 60.0, -1.70, -0.50),
    ]
    spec += [(pid, HOME, False, x, y, vx, vy) for pid, x, y, vx, vy in home_out]
    spec += [(pid, AWAY, False, x, y, vx, vy) for pid, x, y, vx, vy in away_out]

    def _at(pid: int, t: float) -> tuple[float, float]:
        for p, _team, _gk, x, y, vx, vy in spec:
            if p == pid:
                return (x + vx * (t - t_ref), y + vy * (t - t_ref))
        raise KeyError(pid)

    # The ball FOLLOWS THE PLAY: it is at the acting player's feet at each action's timestamp and
    # travels between them. Not decoration -- `infer_ball_carrier` needs a player within
    # `tolerance_m` of the ball to name a carrier, `derive_team_in_possession` needs that carrier,
    # and `add_das` needs possession. A ball on its own constant velocity diverges from every
    # player within ~0.4 s, after which the carrier is unresolved, possession is NaN, and DAS
    # reports `unscoreable_frame` for every action but the first -- measured, on the first draft
    # of this scene. Intermediate frames may still have no carrier (the ball is genuinely in
    # flight); the frames that must resolve are the ones the actions link to.
    #
    # The ball is deliberately never ON the halfway line: there a point reflection is the identity
    # in x, which silently disarms any pitch-absolute check that samples it -- the witness in
    # test_mirror_registry.py caught exactly that when an earlier draft drifted it onto x=52.5.
    ball_waypoints = [
        (8.0, 10),  # a1: HOME pass, at p10's feet
        (8.4, 17),  # a2: received by p17, who shoots
        (9.0, 60),  # a3: AWAY pass
        (9.4, 63),  # a4: AWAY reception
        (10.0, 13),  # a5: HOME regains
        (10.4, 67),  # a6: AWAY shot
    ]

    def _ball_at(t: float) -> tuple[float, float]:
        """Linear interpolation along the waypoints, clamped-extrapolated at both ends."""
        pts = [(wt, _at(pid, wt)) for wt, pid in ball_waypoints]
        # Index the segment explicitly rather than leaking a loop variable past `break`:
        # the leak is what ruff B007 flags, and it is genuinely fragile -- an empty or
        # single-element `pts` would silently reuse whatever the previous iteration bound.
        last = len(pts) - 2
        i = 0 if t <= pts[0][0] else last if t >= pts[-1][0] else min(last, sum(1 for wt, _ in pts if wt <= t) - 1)
        (t0, p0), (t1, p1) = pts[i], pts[i + 1]
        f = (t - t0) / (t1 - t0)
        return (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]))

    # Frame grid spans the action span so every action links to a real frame.
    frame_times = [round(7.6 + 0.2 * i, 2) for i in range(16)]  # 7.6 .. 10.6
    recs = []
    for frame_id, t in enumerate(frame_times, start=100):
        for pid, team, gk, _x, _y, vx, vy in spec:
            px, py = _at(pid, t)
            rec = {
                **_BASE_FRAME,
                "frame_id": frame_id,
                "time_seconds": t,
                "player_id": pid,
                "team_id": team,
                "is_goalkeeper": gk,
                "x": px,
                "y": py,
                "vx": vx,
                "vy": vy,
                "speed": float(np.hypot(vx, vy)),
                "team_attacking_direction": "ltr" if team == HOME else "rtl",
                "is_ball": False,
            }
            recs.append(rec)
        bx, by = _ball_at(t)
        # Ball velocity is the SECANT of its own path, so the declared columns and the observed
        # displacement agree for the ball exactly as they do for the players.
        bx_prev, by_prev = _ball_at(t - 0.2)
        bvx, bvy = (bx - bx_prev) / 0.2, (by - by_prev) / 0.2
        assert abs(bx - 52.5) > 1e-6, f"ball landed on the halfway line at t={t}"
        recs.append(
            {
                **_BASE_FRAME,
                "frame_id": frame_id,
                "time_seconds": t,
                "player_id": np.nan,
                "team_id": np.nan,
                "is_goalkeeper": False,
                "x": bx,
                "y": by,
                "vx": bvx,
                "vy": bvy,
                "speed": float(np.hypot(bvx, bvy)),
                "team_attacking_direction": None,
                "is_ball": True,
            }
        )
    frames = pd.DataFrame(recs)

    pass_id = spadlconfig.actiontype_id["pass"]
    shot_id = spadlconfig.actiontype_id["shot"]

    def _home(aid, pid, type_id, t, end_xy):
        sx, sy = _at(pid, t)
        return dict(
            action_id=aid,
            team_id=HOME,
            player_id=float(pid),
            type_id=type_id,
            time_seconds=t,
            start_x=sx,
            start_y=sy,
            end_x=end_xy[0],
            end_y=end_xy[1],
        )

    def _away(aid, pid, type_id, t, end_xy):
        # AWAY actions are action-LTR == the POINT REFLECTION of the frame position.
        fx, fy = _at(pid, t)
        return dict(
            action_id=aid,
            team_id=AWAY,
            player_id=float(pid),
            type_id=type_id,
            time_seconds=t,
            start_x=FIELD_LENGTH - fx,
            start_y=FIELD_WIDTH - fy,
            end_x=end_xy[0],
            end_y=end_xy[1],
        )

    # HOME pass -> the receiver SHOOTS: a reception that is itself a same-team shot decides
    # `packing_secured` True immediately, and the pass travels PAST away defenders so it has a
    # bypass line at all (a pass that bypasses nobody has NaN line_x and is <NA> by definition).
    a1_end = _at(17, 8.0)
    a2_start_t = 8.4
    # AWAY pass -> HOME touch: an opponent possession boundary decides False, giving the column
    # TWO distinct values rather than a constant.
    a3_end = _at(63, 9.0)

    # Two DECIDED `packing_secured` rows, by the two distinct mechanisms the label supports --
    # a constant column would satisfy "non-NA" while proving nothing:
    #   a1 (HOME pass) -> a2 is the receiver's SHOT            -> True  (decided immediately)
    #   a3 (AWAY pass) -> a4 is a same-team reception, then a5 -> False (opponent boundary)
    # An AWAY pass whose NEXT touch is already an opponent has no reception at all, so its
    # receiver is unresolved and the row is <NA> by definition -- which is why a4 must be AWAY.
    acts = [
        _home(1, 10, pass_id, 8.0, a1_end),
        _home(2, 17, shot_id, a2_start_t, (105.0, 34.0)),
        _away(3, 60, pass_id, 9.0, (FIELD_LENGTH - a3_end[0], FIELD_WIDTH - a3_end[1])),
        _away(4, 63, pass_id, 9.4, _at(66, 9.4)),
        _home(5, 13, pass_id, 10.0, _at(14, 10.0)),
        _away(6, 67, shot_id, 10.4, (105.0, 34.0)),
    ]
    actions = pd.DataFrame(
        [
            {
                **a,
                "game_id": 1,
                "period_id": 1,
                "result_id": 1,
                "bodypart_id": spadlconfig.bodypart_id["foot"],
            }
            for a in acts
        ]
    )
    return actions, frames


def mirror_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Physical mirror: point-reflect positions, NEGATE velocities, swap direction labels.

    Velocities NEGATE rather than reflect (ADR-045): a point reflection maps a vector to its
    negation. Omitting that was live defect D1 in ADR-045, so it is done here explicitly.
    """
    f = frames.copy()
    f["x"] = FIELD_LENGTH - f["x"]
    f["y"] = FIELD_WIDTH - f["y"]
    for vcol in ("vx", "vy"):
        if vcol in f.columns:
            f[vcol] = -f[vcol]
    f["team_attacking_direction"] = f["team_attacking_direction"].map({"ltr": "rtl", "rtl": "ltr"})
    return f


def away_mask(actions: pd.DataFrame, home_team_id) -> np.ndarray:
    from silly_kicks.id_compat import ids_match

    return (~ids_match(actions["team_id"], home_team_id)).to_numpy(dtype=bool)


@functools.cache
def gate_xt():
    """A NON-degenerate xT for the gate. Deliberately y-ASYMMETRIC.

    A y-symmetric grid cannot distinguish a correct point reflection from an x-only mirror --
    exactly the blind spot that let ADR-041's incomplete repair through. Do NOT "simplify" this
    to a pure x-ramp.
    """
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    x_ramp = np.linspace(0.02, 0.9, 16)[None, :]
    y_tilt = np.linspace(0.6, 1.4, 12)[:, None]
    xt.xT = x_ramp * y_tilt
    return xt


def _load_group_modules() -> None:
    """Import every ``_mirror_entries`` module so its ``register()`` populates the registry.

    Auto-discovery rather than a hand-written import list: a manifest is one more thing that can
    go stale, and the whole point of this registry is that a surface cannot drift out of it
    silently. A group module that fails to import is a HARD error -- swallowing it would silently
    shrink the gate's coverage, which is the exact failure mode the meta-assertions exist to catch.
    """
    import importlib
    import pkgutil

    from tests.tracking import _mirror_entries

    for mod in pkgutil.iter_modules(_mirror_entries.__path__):
        module = importlib.import_module(f"tests.tracking._mirror_entries.{mod.name}")
        module.register()


_load_group_modules()
