"""reflect_corners_to_canonical_side -- Y-normalise every corner to one flag (TF-46, spec §8).

The two corner flags sit at opposite ``y`` within the frame-LTR frame (both on the attacking goal line,
so ``x`` is unchanged). Canonicalising the side is therefore a pure Y-reflection of the non-canonical
corners, applied through the shared ``silly_kicks.reflection`` machinery. Reflection runs BEFORE zone
assignment so the GMMs are fit/served in one frame.

``CORNER_REFLECTION_KINDS`` DECLARES the kind of every emitted corner geometry column; a completeness
meta-assertion (``tests/corners/test_corner_orientation_invariance.py``) pins that every geometry-named
output column is declared, carrying the declaration-completeness property that the ``tracking.__all__``-
pinned ``test_mirror_registry`` Gate A/B provides for tracking aggregators (this family is not in
``tracking.__all__``).
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.reflection import ReflectionKind, reflect_columns

#: The kind of every emitted corner geometry column. ``x`` is invariant under the side flip (both
#: flags share the attacking goal line); ``y`` reflects; a y-velocity (none emitted today) would be
#: ``vector_y``. Keep this complete -- the completeness gate reds if a geometry column is missing.
CORNER_REFLECTION_KINDS: dict[str, ReflectionKind] = {
    "kick_x": "invariant",
    "kick_y": "point_y",
}

_CANONICAL_SIDE = "left"


def reflect_corners_to_canonical_side(
    df: pd.DataFrame,
    *,
    kinds: dict[str, ReflectionKind] = CORNER_REFLECTION_KINDS,
    canonical_side: str = _CANONICAL_SIDE,
    side_column: str = "corner_side",
) -> pd.DataFrame:
    """Reflect every non-canonical-side corner's geometry to the canonical flag. Pure (new frame).

    Parameters
    ----------
    df: a corner table (or a per-corner geometry frame) carrying ``side_column``.
    kinds: the corner reflection-kinds map.
    canonical_side: the flag everything is normalised to.
    side_column: the column holding each row's ``left`` / ``right`` flag.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.corners import reflect_corners_to_canonical_side
    >>> df = pd.DataFrame({"corner_side": ["right"], "kick_x": [104.0], "kick_y": [8.0]})
    >>> out = reflect_corners_to_canonical_side(df)
    >>> float(out.loc[0, "kick_y"])  # 68 - 8
    60.0
    """
    if side_column not in df.columns:
        raise KeyError(f"reflect_corners_to_canonical_side: {side_column!r} not in df.")
    mask = (df[side_column] != canonical_side).to_numpy()
    point_x = [c for c, k in kinds.items() if k == "point_x"]
    point_y = [c for c, k in kinds.items() if k == "point_y"]
    vector_x = [c for c, k in kinds.items() if k == "vector_x"]
    vector_y = [c for c, k in kinds.items() if k == "vector_y"]
    return reflect_columns(
        df,
        mask,
        point_x=point_x,
        point_y=point_y,
        vector_x=vector_x,
        vector_y=vector_y,
    )
