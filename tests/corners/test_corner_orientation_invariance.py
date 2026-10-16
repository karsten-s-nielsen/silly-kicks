"""TF-46 dedicated orientation-invariance gate (replaces the tracking.__all__-pinned mirror Gate A/B).

(a) behavioural: a left-corner scene and its mirrored right-corner twin of identical geometry reflect to
    equal canonical geometry; non-vacuity probe: the right row MUST move.
(b) declaration-completeness (P2): every emitted corner geometry column (CORNER_*_COLUMNS matching
    reflection.GEOMETRIC_NAME) is declared in CORNER_REFLECTION_KINDS -- so a future geometry column
    cannot silently escape reflection.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.corners import (
    CORNER_CHANCE_COLUMNS,
    CORNER_IDENTIFY_COLUMNS,
    CORNER_WINDOW_COLUMNS,
    reflect_corners_to_canonical_side,
)
from silly_kicks.corners._reflect import CORNER_REFLECTION_KINDS
from silly_kicks.reflection import GEOMETRIC_NAME


def test_mirror_twins_reflect_to_equal_geometry():
    # kick_y 60 (left) and 8 (right) are mirror images about the y=34 midline (68-60 == 8).
    df = pd.DataFrame({"corner_side": ["left", "right"], "kick_x": [104.0, 104.0], "kick_y": [60.0, 8.0]})
    out = reflect_corners_to_canonical_side(df)
    assert out.loc[0, "kick_y"] == out.loc[1, "kick_y"]  # mirror-equal after canonicalise


def test_non_vacuity_right_row_moves():
    df = pd.DataFrame({"corner_side": ["right"], "kick_x": [104.0], "kick_y": [8.0]})
    out = reflect_corners_to_canonical_side(df)
    assert out.loc[0, "kick_y"] != 8.0  # it MUST have moved


def test_every_emitted_geometry_column_is_declared():
    emitted = set(CORNER_IDENTIFY_COLUMNS) | set(CORNER_WINDOW_COLUMNS) | set(CORNER_CHANCE_COLUMNS)
    geometry = {c for c in emitted if GEOMETRIC_NAME.match(c)}
    undeclared = geometry - set(CORNER_REFLECTION_KINDS)
    assert not undeclared, f"geometry columns not declared in CORNER_REFLECTION_KINDS: {sorted(undeclared)}"
    # non-vacuity: the enumeration actually found the known geometry columns
    assert {"kick_x", "kick_y"} <= geometry
