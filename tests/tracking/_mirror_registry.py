"""Registry backing the ADR-028 mirror gates (spec 2026-07-29, section 6).

TWO gates, because one instrument cannot see both defect classes:

* **Gate A** -- physical mirror. Detects CONVENTION MIXING (an action-LTR value combined with a
  frame-LTR one). ``home_team_id`` is swapped, because after a physical mirror the team attacking
  +x really is the other one.
* **Gate B** -- ``home_team_id`` invariance on FIXED canonical frames. Detects IDENTITY-KEYED
  direction inference, which Gate A is structurally BLIND to: swapping ``home_team_id`` restores
  the very invariant identity-keying assumes, so an identity-keyed aggregator is invariant under
  Gate A whether it is safe or not.

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
) -> None:
    MIRROR_ENTRIES[name] = MirrorEntry(
        name=name,
        call=call,
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
    """``(actions, frames)`` -- canonical converter shape, DELIBERATELY y-ASYMMETRIC.

    y-asymmetry is not decoration: an x-only reprojection is exact on a y-symmetric scene, and
    ADR-041 shipped precisely that incomplete repair -- only a y-asymmetric oracle caught it.

    Home attacks +x (``"ltr"``); away attacks -x (``"rtl"``). BOTH teams act, so the away rows --
    the only rows an ADR-028 defect touches -- are a real population rather than a single token
    action. Three frames give the pre-window and velocity-derivative features some history.
    """
    from silly_kicks.spadl import config as spadlconfig

    rows: list[dict] = [
        dict(player_id=1, team_id=HOME, is_goalkeeper=True, x=5.0, y=27.0, d="ltr"),
        dict(player_id=50, team_id=AWAY, is_goalkeeper=True, x=100.0, y=41.0, d="rtl"),
    ]
    # y values spread ASYMMETRICALLY about y=34 on purpose.
    for i, (x, y) in enumerate(
        [(28, 12), (36, 21), (44, 9), (52, 30), (60, 17), (33, 44), (47, 55), (58, 38), (25, 50), (41, 62)]
    ):
        rows.append(dict(player_id=10 + i, team_id=HOME, is_goalkeeper=False, x=float(x), y=float(y), d="ltr"))
    for i, (x, y) in enumerate(
        [(70, 14), (63, 25), (77, 8), (55, 33), (68, 19), (74, 47), (61, 58), (80, 36), (66, 52), (50, 60)]
    ):
        rows.append(dict(player_id=60 + i, team_id=AWAY, is_goalkeeper=False, x=float(x), y=float(y), d="rtl"))
    # The ball sits deliberately OFF BOTH centre lines. On the halfway line a point reflection is
    # the identity in x, which silently disarms any pitch-absolute check that samples the ball --
    # the witness in test_mirror_registry.py caught exactly that when an earlier draft of this
    # fixture drifted the ball onto x=52.5.
    rows.append(dict(player_id=np.nan, team_id=np.nan, is_goalkeeper=False, x=38.0, y=23.0, d=None))

    recs = []
    for frame_id, t in ((100, 7.6), (101, 7.8), (102, 8.0)):
        for r in rows:
            rec = {**_BASE_FRAME, **r, "frame_id": frame_id, "time_seconds": t}
            rec["team_attacking_direction"] = rec.pop("d")
            rec["is_ball"] = bool(pd.isna(rec["team_id"]))
            # Positions are held CONSTANT across the three frames and velocity is stated
            # explicitly. A positional drift would desynchronise the action anchors below (which
            # name a specific frame position) from the frame the linker actually picks, for no
            # gain: mirror invariance is about the two LEGS agreeing, not about motion.
            rec["vx"], rec["vy"] = 0.8, -0.5
            recs.append(rec)
    frames = pd.DataFrame(recs)

    pass_id = spadlconfig.actiontype_id["pass"]
    shot_id = spadlconfig.actiontype_id["shot"]
    acts = [
        # HOME actions: action-LTR == frame coords.
        dict(
            action_id=1,
            team_id=HOME,
            player_id=10.0,
            type_id=pass_id,
            start_x=28.0,
            start_y=12.0,
            end_x=36.0,
            end_y=21.0,
        ),
        dict(
            action_id=2,
            team_id=HOME,
            player_id=13.0,
            type_id=shot_id,
            start_x=52.0,
            start_y=30.0,
            end_x=105.0,
            end_y=34.0,
        ),
        # AWAY actions: action-LTR == the POINT REFLECTION of the frame position.
        dict(
            action_id=3,
            team_id=AWAY,
            player_id=60.0,
            type_id=pass_id,
            start_x=FIELD_LENGTH - 70.0,
            start_y=FIELD_WIDTH - 14.0,
            end_x=FIELD_LENGTH - 63.0,
            end_y=FIELD_WIDTH - 25.0,
        ),
        dict(
            action_id=4,
            team_id=AWAY,
            player_id=63.0,
            type_id=shot_id,
            start_x=FIELD_LENGTH - 55.0,
            start_y=FIELD_WIDTH - 33.0,
            end_x=105.0,
            end_y=34.0,
        ),
    ]
    actions = pd.DataFrame(
        [
            {
                **a,
                "game_id": 1,
                "period_id": 1,
                "result_id": 1,
                "time_seconds": 8.0,
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
