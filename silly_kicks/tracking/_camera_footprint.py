"""SkillCorner camera-footprint geometry (TF-46 Arm 1 Part 2, spec section 17.7 / CL-15, DL ruling).

The footprint is the four image-projection corners of the broadcast camera view, in the SPADL frame. A point
is "in camera view" when it lies inside that quadrilateral (within a metre tolerance). The quadrilateral is
convex for a planar camera projection; a non-convex or zero-area quad is degenerate (a near-horizon
projection) and yields ``<NA>`` -- never a false in/out.

Pure numpy / pandas. The corners column order (tl, tr, br, bl) is NOT assumed: the point-in-quad test accepts
either winding (all cross products non-negative OR all non-positive), so a reflected footprint tests the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..reflection import reflect_columns
from ._provider_visibility import detected_mask

#: The four footprint corner column stems, in storage order (top-left, top-right, bottom-right, bottom-left).
_CORNERS = ("tl", "tr", "br", "bl")
FOOTPRINT_X_COLUMNS = tuple(f"cam_{c}_x" for c in _CORNERS)
FOOTPRINT_Y_COLUMNS = tuple(f"cam_{c}_y" for c in _CORNERS)
_AREA_EPS = 1e-6  # m^2: below this the quad is treated as zero-area (degenerate)


def _corner_array(footprint_rows: pd.DataFrame) -> np.ndarray:
    """Stack the footprint columns into an ``(N, 4, 2)`` float64 array (tl, tr, br, bl)."""
    xs = np.stack([footprint_rows[c].to_numpy(dtype="float64") for c in FOOTPRINT_X_COLUMNS], axis=1)
    ys = np.stack([footprint_rows[c].to_numpy(dtype="float64") for c in FOOTPRINT_Y_COLUMNS], axis=1)
    return np.stack([xs, ys], axis=2)  # (N, 4, 2)


def _signed_area(c: np.ndarray) -> np.ndarray:
    """Twice the signed shoelace area per quad (sign = winding)."""
    a = np.zeros(len(c))
    for i in range(4):
        j = (i + 1) % 4
        a += c[:, i, 0] * c[:, j, 1] - c[:, j, 0] * c[:, i, 1]
    return a


def _degenerate(c: np.ndarray) -> np.ndarray:
    """Per quad: True when NaN, zero-area, or non-convex (consecutive turns change sign)."""
    finite = np.isfinite(c).all(axis=(1, 2))
    area = _signed_area(c)
    zero_area = np.abs(area) < 2 * _AREA_EPS
    # Convexity: every consecutive edge pair turns the same way.
    turns = np.zeros((len(c), 4))
    for i in range(4):
        j, k = (i + 1) % 4, (i + 2) % 4
        e1x, e1y = c[:, j, 0] - c[:, i, 0], c[:, j, 1] - c[:, i, 1]
        e2x, e2y = c[:, k, 0] - c[:, j, 0], c[:, k, 1] - c[:, j, 1]
        turns[:, i] = e1x * e2y - e1y * e2x
    pos = (turns > 0).sum(axis=1)
    neg = (turns < 0).sum(axis=1)
    nonconvex = (pos > 0) & (neg > 0)  # a mix of left and right turns
    return (~finite) | zero_area | nonconvex


def in_camera_view_mask(x, y, corners: np.ndarray, *, tol_m: float = 0.0) -> pd.arrays.BooleanArray:
    """Nullable boolean: is ``(x, y)`` inside its quad ``corners`` (``(N, 4, 2)``), within ``tol_m`` metres?

    ``<NA>`` where the quad is degenerate (NaN / zero-area / non-convex) or the point's own position is NaN.
    Corner-order-independent: the interior side is taken from the quad's own winding, so a reflected footprint
    gives the same mask. A point up to ``tol_m`` OUTSIDE an edge still counts inside (edge points count inside).
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    c = np.asarray(corners, dtype="float64")
    n = len(x)
    degen = _degenerate(c)
    wind = np.sign(_signed_area(c))
    wind[wind == 0] = 1.0
    inside = np.ones(n, dtype=bool)
    for i in range(4):
        j = (i + 1) % 4
        ex, ey = c[:, j, 0] - c[:, i, 0], c[:, j, 1] - c[:, i, 1]
        px, py = x - c[:, i, 0], y - c[:, i, 1]
        cross = ex * py - ey * px  # +interior for ccw winding
        length = np.hypot(ex, ey)
        signed_dist = np.where(length > 0, (cross * wind) / np.where(length > 0, length, 1.0), -np.inf)
        inside &= signed_dist >= -tol_m
    valid = (~degen) & np.isfinite(x) & np.isfinite(y)
    out = pd.array([pd.NA] * n, dtype="boolean")
    out[valid] = inside[valid]
    return out


def footprint_degenerate_mask(footprint_rows: pd.DataFrame) -> np.ndarray:
    """Per footprint row: True when its quad is degenerate (NaN / zero-area / non-convex)."""
    return _degenerate(_corner_array(footprint_rows))


def orient_camera_footprint(footprint: pd.DataFrame, flips) -> pd.DataFrame:
    """Re-orient a footprint table by an EXPLICIT per-(game_id, period_id) flip mapping (spec section 17.7).

    ``flips`` maps ``(game_id, period_id) -> True | False | None``; ``True`` point-reflects that (game, period)'s
    footprint corners (105 - x, 68 - y) via ``reflect_columns`` (ADR-045 point pairs, float32-safe). Frames carry
    no recoverable flip record, so a consumer that re-orients frames passes the SAME mask it used on the frames.
    Pure: returns a new frame.
    """
    keys = list(zip(footprint["game_id"], footprint["period_id"], strict=True))
    mask = np.array([bool(flips.get((g, int(p))) or False) for g, p in keys], dtype=bool)
    return reflect_columns(
        footprint,
        mask,
        point_x=list(FOOTPRINT_X_COLUMNS),
        point_y=list(FOOTPRINT_Y_COLUMNS),
        field_length=105.0,
        field_width=68.0,
    )


def footprint_inside_share(frames: pd.DataFrame, footprint: pd.DataFrame, *, tol_m: float = 1.0) -> float:
    """Share of DETECTED object rows (``detected_mask``) lying inside their frame's footprint within ``tol_m``
    (the spec section 16.1 validation gate). Rows whose footprint is null / degenerate are excluded from the
    denominator (no footprint to test against). NaN when no evaluable detected row."""
    fp = footprint.set_index(["game_id", "period_id", "frame_id"])
    det = detected_mask(frames["visibility"], provider="skillcorner")
    obj = frames.loc[det & ~frames["is_ball"].astype(bool)]
    if obj.empty:
        return float("nan")
    keys = list(zip(obj["game_id"], obj["period_id"].astype(int), obj["frame_id"].astype(int), strict=True))
    present = [k in fp.index for k in keys]
    rows = fp.reindex([k for k, ok in zip(keys, present, strict=True) if ok])
    if rows.empty:
        return float("nan")
    corners = _corner_array(rows)
    sub = obj.loc[present]
    mask = in_camera_view_mask(sub["x"].to_numpy("float64"), sub["y"].to_numpy("float64"), corners, tol_m=tol_m)
    evaluable = ~mask.isna()
    if not bool(evaluable.any()):
        return float("nan")
    return float(np.asarray(mask[evaluable], dtype=bool).mean())


def keeper_undetected_split(frames: pd.DataFrame) -> dict[str, int | float]:
    """Counts and shares of GOALKEEPER object-frames that are undetected IN camera view vs OFF camera (R4,
    B-22): the evidence a later owner decision on wiring ``in_camera_view`` into the GK detection gate needs.

    ``in_view`` / ``off_camera`` count keeper frames with ``in_camera_view`` True / False among the UNDETECTED
    (``~detected_mask``); ``unknown_view`` counts those whose ``in_camera_view`` is ``<NA>``. Shares are over the
    undetected keeper frames.
    """
    gk = frames.loc[frames["is_goalkeeper"].astype(bool) & ~frames["is_ball"].astype(bool)]
    det = detected_mask(gk["visibility"], provider="skillcorner")
    und = gk.loc[~det]
    n = len(und)
    icv = und["in_camera_view"]
    in_view = int((icv == True).sum())  # noqa: E712 (nullable boolean: `== True` excludes <NA>)
    off_camera = int((icv == False).sum())  # noqa: E712
    unknown = int(icv.isna().sum())
    return {
        "n_undetected_keeper_frames": n,
        "in_view": in_view,
        "off_camera": off_camera,
        "unknown_view": unknown,
        "in_view_share": (in_view / n) if n else float("nan"),
        "off_camera_share": (off_camera / n) if n else float("nan"),
    }
