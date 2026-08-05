"""Derive a column's visible-area applicability class by perturbation.

Both probes move player POSITIONS at fixed roster and fixed polygon, so neither collapses into
the other. Masking by polygon would BE roster variation and would discriminate nothing.

The class is DERIVED, not declared. A human picking one of three categories would put a
declaration inside the LOCKED half of the registry -- the observation/adjudication conflation
this design exists to prevent.

Spec: docs/superpowers/specs/2026-08-04-sb360-coverage-audit-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.sb360 import _fixture as F
from tests.sb360._registry import Sb360Entry

#: Relocation targets. Both probes move ONE player to the pitch corner farthest from that
#: frame's ball; only WHICH player differs. Symmetric, deterministic, and never clamped.
_PITCH_CORNERS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.0, 68.0),
    (105.0, 0.0),
    (105.0, 68.0),
)

#: Movement below this is float noise, not a response.
_MOVED_DELTA = 1e-9


def _value(entry: Sb360Entry, column: str, frames: pd.DataFrame, actions, links) -> float:
    """Scalar summary of a column, robust to NaN AND to non-float dtypes.

    Routed through nullable ``Float64`` before going to numpy: a plain ``fillna(0.0)`` raises
    ``TypeError: Invalid value '0.0' for dtype 'boolean'`` on a nullable-boolean column such as
    ``packing_secured``. Non-numeric columns coerce to all-NaN and summarise to 0.0, which is
    correct for a probe -- an unmoved string column is genuinely unmoved.
    """
    out = entry.call(actions, frames, links, F.HOME_TEAM_ID)
    numeric = pd.to_numeric(out[column], errors="coerce").astype("Float64")
    return float(np.nansum(numeric.to_numpy(dtype="float64", na_value=np.nan)))


def _shift(frames: pd.DataFrame, *, extreme: bool) -> pd.DataFrame:
    """Relocate ONE player to the pitch corner farthest from the ball in its own frame.

    Distance is measured to the BALL, not to the players' centroid. A region-support feature
    queries around the ACTION, so "far from the action, beyond any plausible query radius" is
    the property that matters. A centroid-relative probe was tried and measured wrong: this
    fixture's player centroid sits at x=65.6 while the balls sit elsewhere, so the "nearest"
    player was 12.5 m outside a 15 m query band and neither probe moved a genuine region
    feature at all.

    Relocation rather than a fixed radial displacement, for the same measured reason: the
    player nearest a goal-kick sits ~1 m from a ball on the goal line, so an 8 m outward push
    CLAMPS at the touchline, lands 4 m away, and never leaves the query band. A corner is
    deterministic, always far from the ball, and cannot clamp.

    Only WHICH player differs between the probes -- ``extreme`` takes the farthest from the
    ball (a hull's defining member; a fixed region never contained it), ``near`` takes the
    closest (inside any plausible region, so a region feature must notice its departure).
    """
    out = frames.copy()
    is_player = ~out["is_ball"].astype(bool)
    players = out[is_player]
    balls = out[~is_player]
    if players.empty or balls.empty:
        return out

    ball_xy = balls.set_index("frame_id")[["x", "y"]].astype(float)
    bx = players["frame_id"].map(ball_xy["x"])
    by = players["frame_id"].map(ball_xy["y"])
    dx = players["x"].astype(float) - bx
    dy = players["y"].astype(float) - by
    dist = np.hypot(dx.to_numpy(), dy.to_numpy())
    dist_s = pd.Series(dist, index=players.index)

    target = dist_s.idxmax() if extreme else dist_s.idxmin()

    # Relocate to the pitch CORNER farthest from that frame's ball, rather than displacing by a
    # fixed radial distance. A radial push is fragile exactly where it matters: the nearest
    # player to a goal-kick sits ~1 m from a ball on the goal line, so an 8 m outward push
    # CLAMPS at the touchline, lands 4 m away, and never leaves a 15 m query band -- measured.
    # A corner is deterministic, always far from the ball, and cannot clamp.
    b_x = float(bx.loc[target])
    b_y = float(by.loc[target])
    corner = max(_PITCH_CORNERS, key=lambda c: (c[0] - b_x) ** 2 + (c[1] - b_y) ** 2)
    out.loc[target, "x"] = corner[0]
    out.loc[target, "y"] = corner[1]
    return out


def derive_applicability(entry: Sb360Entry, column: str) -> tuple[str, dict[str, float]]:
    """Return ``(applicability_class, {"extreme": delta, "near": delta})``.

    Probe 1 runs FIRST and WINS. A feature can satisfy both, and data-defined support is the
    dangerous property: it is the one where a coverage fraction reads as reassurance while
    being circular, because a hull over visible players is 100% observed by construction.
    """
    actions, frames, links = F.build_leg_a()
    base = _value(entry, column, frames, actions, links)

    extreme_delta = abs(_value(entry, column, _shift(frames, extreme=True), actions, links) - base)
    near_delta = abs(_value(entry, column, _shift(frames, extreme=False), actions, links) - base)
    deltas = {"extreme": extreme_delta, "near": near_delta}

    if extreme_delta > _MOVED_DELTA:
        return "support_data_defined", deltas
    if near_delta > _MOVED_DELTA:
        return "region_support", deltas
    return "no_support", deltas
