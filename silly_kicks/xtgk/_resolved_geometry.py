"""Resolved GK-distribution geometry (ADR-036 amendment, 4.46.0).

The GK-distribution domain's canonical SPADL coords are NOT trustworthy:

* **Gradient Sports** -- ~60% of goal-kicks carry a NaN origin (the taker is not in the raw event).
* **SkillCorner** -- the native goal-kick origin is the broadcast BALL detection, not the keeper
  (ADR-024 / PR-S104): PRESENT, finite, and ~10-20 m wrong.

v1 (``tracking/_xt_gk.py``) resolves both via ``resolve_gk_geometry``, and the lakehouse persists the
result as ``fct_action_context.xt_gk_origin_x/_y`` + ``xt_gk_dest_x/_y`` (PR-S101). v2 never read
them. This module is the ONE callable that injects them, so the rule lives in a single place rather
than as a prose contract two consumers must each re-derive -- which is exactly how the bug happened.

Policy lives at the EDGE: the metric engine stays provenance-free and reads exactly
``start_x``/``end_x``. This is a **transient scoring-time view** -- canonical coordinates are never
written back, so ADR-025's never-mutate-canonical fence is intact.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

GK_GEOMETRY_SOURCE_COLUMN = "gk_geometry_source"

#: The seven stamp values. ``unresolved`` WINS over any ``resolved_*``/``native`` when a coordinate
#: is still non-finite -- it answers "will this row score?", pairing with the metric's NaN guard.
GK_GEOMETRY_SOURCES = (
    "off_domain",
    "native",
    "resolved_origin",
    "resolved_dest",
    "resolved_both",
    "unresolved",
    "unattested",
)


def _changed(raw: np.ndarray, res: np.ndarray) -> np.ndarray:
    """True where the resolved value EXISTS and differs from the raw one (a NaN raw counts as a
    difference, so a rescue registers as a change)."""
    return np.isfinite(res) & ~np.isclose(raw, res, atol=1e-9, rtol=0.0, equal_nan=True)


def apply_resolved_gk_geometry(
    actions: pd.DataFrame,
    *,
    domain_column: str = "is_gk_distribution",
    origin_columns: tuple[str, str] = ("xt_gk_origin_x", "xt_gk_origin_y"),
    dest_columns: tuple[str, str] = ("xt_gk_dest_x", "xt_gk_dest_y"),
) -> pd.DataFrame:
    """OVERRIDE the GK-distribution rows' coords with gold's resolved keeper geometry; stamp provenance.

    PURE: returns a NEW frame, never mutates ``actions``.

    **Override, not coalesce.** A ``fillna`` would rescue Gradient Sports' NaN origins and silently
    leave SkillCorner's *present-and-wrong* broadcast-ball origins in place.

    Parameters
    ----------
    actions : pd.DataFrame
        Attack-LTR SPADL with ``start_x``/``start_y``/``end_x``/``end_y`` and ``domain_column``.
    domain_column : str
        The GK-distribution domain flag. **Required** -- absent raises, because treating every row
        as in-domain would overwrite open-play coordinates with keeper geometry.
    origin_columns, dest_columns : tuple[str, str]
        The resolved coordinates. If any are absent this is an observable no-op (warn) and every
        in-domain row is stamped ``unattested`` -- never ``native``, which would suppress the
        metric's warn-once while the origins were still raw.

    Returns
    -------
    pd.DataFrame
        A copy with the overridden coordinates plus a ``gk_geometry_source`` column.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(
    ...     {"is_gk_distribution": [True], "start_x": [25.0], "start_y": [40.0],
    ...      "end_x": [40.0], "end_y": [34.0], "xt_gk_origin_x": [4.29],
    ...      "xt_gk_origin_y": [34.0], "xt_gk_dest_x": [40.0], "xt_gk_dest_y": [34.0]}
    ... )
    >>> out = apply_resolved_gk_geometry(a)
    >>> float(out.loc[0, "start_x"]), out.loc[0, "gk_geometry_source"]
    (4.29, 'resolved_origin')
    """
    if domain_column not in actions.columns:
        raise ValueError(
            f"apply_resolved_gk_geometry requires the domain column {domain_column!r}. Without it "
            "every row would be treated as a GK distribution and open-play coordinates would be "
            "overwritten with keeper geometry. Supply fct_action_context.is_gk_distribution."
        )

    out = actions.copy()
    domain = out[domain_column].fillna(False).to_numpy(dtype=bool)
    source = np.where(domain, "unattested", "off_domain").astype(object)

    ox, oy = origin_columns
    dx_c, dy_c = dest_columns
    missing = [c for c in (ox, oy, dx_c, dy_c) if c not in out.columns]
    if missing:
        warnings.warn(
            f"apply_resolved_gk_geometry: resolved-coordinate columns {missing} are absent -- "
            "no-op; GK-distribution origins remain RAW (Gradient Sports NaN / SkillCorner "
            "broadcast-ball). Rows stamped 'unattested'. Add xt_gk_origin_x/_y + xt_gk_dest_x/_y "
            "to the fct_action_context projection (silly-kicks >= 4.36.0).",
            stacklevel=2,
        )
        out[GK_GEOMETRY_SOURCE_COLUMN] = source
        return out

    def _num(col: str) -> np.ndarray:
        return pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)

    sx, sy, ex, ey = _num("start_x"), _num("start_y"), _num("end_x"), _num("end_y")
    rox, roy, rdx, rdy = _num(ox), _num(oy), _num(dx_c), _num(dy_c)

    origin_changed = domain & (_changed(sx, rox) | _changed(sy, roy))
    dest_changed = domain & (_changed(ex, rdx) | _changed(ey, rdy))

    # Apply wherever a resolved value exists (not only where it differs) -- idempotent.
    apply_o = domain & np.isfinite(rox) & np.isfinite(roy)
    apply_d = domain & np.isfinite(rdx) & np.isfinite(rdy)
    sx = np.where(apply_o, rox, sx)
    sy = np.where(apply_o, roy, sy)
    ex = np.where(apply_d, rdx, ex)
    ey = np.where(apply_d, rdy, ey)
    out["start_x"], out["start_y"], out["end_x"], out["end_y"] = sx, sy, ex, ey

    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(ex) & np.isfinite(ey)
    both = origin_changed & dest_changed
    # S2: a row whose resolved coords are ALL null was never attested by the mart. Stamping it
    # `native` would assert "raw already equalled resolved" -- false, nothing attested it -- and
    # would suppress the metric's warn-once. Such rows keep the initial `unattested`.
    attested = np.isfinite(rox) | np.isfinite(roy) | np.isfinite(rdx) | np.isfinite(rdy)
    source = np.where(domain & attested & origin_changed & ~dest_changed, "resolved_origin", source)
    source = np.where(domain & attested & dest_changed & ~origin_changed, "resolved_dest", source)
    source = np.where(domain & attested & both, "resolved_both", source)
    source = np.where(domain & attested & ~origin_changed & ~dest_changed, "native", source)
    # R3 precedence: `unresolved` wins over every resolved_*/native when a coord is still non-finite.
    source = np.where(domain & ~finite, "unresolved", source)
    out[GK_GEOMETRY_SOURCE_COLUMN] = source
    return out
