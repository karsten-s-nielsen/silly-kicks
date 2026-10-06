"""Vectorised collective-variable and back-line array kernels (TF-58, ADR D12/D13).

Single definition behind ``compute_team_shape`` (centroid, length, width, stretch, hull) and
``compute_defensive_line`` (six back-line columns). The array kernels take numpy arrays only (pure numpy
+ stdlib ``fractions`` for the A2 exact orientation fallback); the ``compute_collective_variables`` pandas
wrapper at the bottom is the one frame-facing entry point. See spec 7.5.

Byte-identity (C1): every reduction runs on a COMPACT ``(R, n)`` slice of rows whose count is exactly
``n``, so numpy applies its 1-D per-row routine (pairwise summation for n >= 8, the default argsort for
ties) exactly as the legacy per-group loop did. A trailing-zero-padded row would not reproduce it.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import restore_id_dtype

COLLECTIVE_VARIABLES: tuple[str, ...] = (
    "centroid_x",
    "centroid_y",
    "team_length",
    "team_width",
    "stretch_index",
    "stretch_x",
    "stretch_y",
    "spread",
    "convex_hull_area",
)
BACK_LINE_VARIABLES: tuple[str, ...] = (
    "defensive_line_x",
    "back_line_high_x",
    "compactness_x",
    "lateral_width",
    "max_lateral_gap",
    "back_n_count",
)

_CCW_ERRBOUND_A = (3.0 + 16.0 * 2.0**-53) * 2.0**-53  # Shewchuk (1997) orient2d static filter bound


def pack_groups(
    codes: np.ndarray, x: np.ndarray, y: np.ndarray, n_groups: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter long-form rows into ``(n_groups, P, 2)`` NaN-padded arrays, valid values LEFT-ALIGNED in
    input row order. Returns ``(pos, counts, first_row)`` where ``first_row[g]`` is the positional index
    of group ``g``'s first input row.

    Examples
    --------
    >>> import numpy as np
    >>> pos, counts, first_row = pack_groups(
    ...     np.array([0, 1, 0]), np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), 2
    ... )
    >>> counts.tolist()
    [2, 1]
    >>> pos[0, :2, 0].tolist()
    [1.0, 3.0]
    """
    order = np.argsort(codes, kind="stable")  # stable: within-group input order preserved
    sc = codes[order]
    counts = np.bincount(codes, minlength=n_groups).astype(np.int64)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    rank = np.arange(sc.size, dtype=np.int64) - starts[sc]
    width = int(counts.max()) if n_groups else 0
    pos = np.full((n_groups, width, 2), np.nan)
    pos[sc, rank, 0] = x[order]
    pos[sc, rank, 1] = y[order]
    first_row = order[starts].astype(np.int64) if n_groups else np.empty(0, dtype=np.int64)
    return pos, counts, first_row


def compact_rows(pos: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left-align the valid slots of each row (slot order preserved). Returns ``(compacted_pos, counts)``.

    Examples
    --------
    >>> import numpy as np
    >>> out, counts = compact_rows(np.array([[[np.nan, np.nan], [1.0, 2.0]]]), np.array([[False, True]]))
    >>> out[0, 0].tolist(), int(counts[0])
    ([1.0, 2.0], 1)
    """
    order = np.argsort(~valid, axis=1, kind="stable")  # valid slots first, slot order kept
    return np.take_along_axis(pos, order[..., None], axis=1), valid.sum(axis=1).astype(np.int64)


def _collinear_exact(pts: np.ndarray) -> bool:
    """Exact rational collinearity of ``(n, 2)`` float points (A2 fallback)."""
    q = [(Fraction(float(px)), Fraction(float(py))) for px, py in pts]
    x0, y0 = q[0]
    ref = next(((px, py) for px, py in q[1:] if (px, py) != (x0, y0)), None)
    if ref is None:
        return True  # all points coincide
    dx, dy = ref[0] - x0, ref[1] - y0
    return all(dx * (py - y0) - dy * (px - x0) == 0 for px, py in q)


def _exactly_collinear(p: np.ndarray) -> np.ndarray:
    """``(R, n, 2)`` float64 -> ``(R,)`` bool, EXACT on the given floats (Shewchuk 1997 adaptive predicate):
    a vectorised float filter settles every row with a certainly non-zero orientation; only the rows it
    cannot settle are re-checked in exact rational arithmetic."""
    r_count = p.shape[0]
    rel = p - p[:, :1, :]  # p_j - p_0 (rounded; the filter bound accounts for it)
    far = np.argmax(np.hypot(rel[..., 0], rel[..., 1]), axis=1)
    v = rel[np.arange(r_count), far]  # p_far - p_0
    det_l = v[:, None, 0] * rel[..., 1]
    det_r = v[:, None, 1] * rel[..., 0]
    certainly_nonzero = np.abs(det_l - det_r) > _CCW_ERRBOUND_A * (np.abs(det_l) + np.abs(det_r))
    out = np.zeros(r_count, dtype=bool)
    for r in np.flatnonzero(~certainly_nonzero.any(axis=1)):  # near-collinear rows only -- rare by construction
        out[r] = _collinear_exact(p[r])
    return out


def _hull_area_fixed_n(p: np.ndarray) -> np.ndarray:
    """Convex-hull area of ``(R, n, 2)`` (n >= 3, all valid) by the angular-gap method (D12): a point is a
    hull vertex iff the largest angular gap between the directions to the other points is >= pi."""
    _, n, _ = p.shape
    # EXACT collinearity (A2) -> exactly 0.0 (the QhullError contract, decided geometrically, not by float == 0.0).
    # Degenerate rows (collinear / coincident) produce inf-inf and 0/0 below; they are masked by `collinear` at the
    # end, so their invalid intermediates are expected and suppressed.
    collinear = _exactly_collinear(p)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = p[:, None, :, :] - p[:, :, None, :]  # d[r, i, j] = p_j - p_i
        coincident = (d[..., 0] == 0.0) & (d[..., 1] == 0.0)  # includes j == i
        ang = np.where(coincident, np.inf, np.arctan2(d[..., 1], d[..., 0]))
        ang.sort(axis=2)  # coincident points carry no direction -> +inf, last
        m = n - coincident.sum(axis=2)  # distinct directions seen from point i
        gaps = np.diff(ang, axis=2)
        k = np.arange(n - 1)
        gaps = np.where(k[None, None, :] < (m - 1)[..., None], gaps, -np.inf)
        last = np.take_along_axis(ang, np.maximum(m - 1, 0)[..., None], axis=2)[..., 0]
        wrap = np.where(m >= 1, ang[..., 0] + 2.0 * np.pi - last, 2.0 * np.pi)
        on_hull = np.maximum(gaps.max(axis=2, initial=-np.inf), wrap) >= np.pi
        # shoelace over hull points ordered by angle around their mean (inside the hull); centred to limit cancellation
        cnt = on_hull.sum(axis=1)
        cx = np.where(on_hull, p[..., 0], 0.0).sum(axis=1) / cnt
        cy = np.where(on_hull, p[..., 1], 0.0).sum(axis=1) / cnt
        qx, qy = p[..., 0] - cx[:, None], p[..., 1] - cy[:, None]
        theta = np.where(on_hull, np.arctan2(qy, qx), np.inf)
        order = np.argsort(theta, axis=1, kind="stable")
        qx, qy = np.take_along_axis(qx, order, axis=1), np.take_along_axis(qy, order, axis=1)
        idx = np.arange(n)[None, :]
        nxt = np.where(idx + 1 < cnt[:, None], idx + 1, 0)
        cross = qx * np.take_along_axis(qy, nxt, axis=1) - np.take_along_axis(qx, nxt, axis=1) * qy
        area = 0.5 * np.abs(np.where(idx < cnt[:, None], cross, 0.0).sum(axis=1))
    return np.where(collinear, 0.0, area)


def hull_area_batch(pos: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Per-row hull area. NaN where count < 3; exactly 0.0 where the points are exactly collinear; else the area.

    Examples
    --------
    >>> import numpy as np
    >>> square = np.array([[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]])
    >>> float(hull_area_batch(square, np.array([4]))[0])
    4.0
    """
    out = np.full(counts.shape[0], np.nan)
    for n in np.unique(counts):
        if n >= 3:
            rows = np.flatnonzero(counts == n)
            out[rows] = _hull_area_fixed_n(pos[rows, :n, :])
    return out


def collective_from_positions(pos: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Collective variables per row (keys == :data:`COLLECTIVE_VARIABLES`); NaN below each variable's
    minimum n (spec 7.5). Reductions run on compact ``(R, n)`` slices (C1 byte-identity).

    Examples
    --------
    >>> import numpy as np
    >>> cv = collective_from_positions(np.array([[[0.0, 0.0], [4.0, 0.0]]]), np.array([2]))
    >>> float(cv["centroid_x"][0]), float(cv["team_length"][0])
    (2.0, 4.0)
    """
    g = counts.shape[0]
    out = {name: np.full(g, np.nan) for name in COLLECTIVE_VARIABLES}
    for n in np.unique(counts):
        if n < 1:
            continue
        rows = np.flatnonzero(counts == n)
        xs = pos[rows, :n, 0]
        ys = pos[rows, :n, 1]
        cx = np.mean(xs, axis=1)
        cy = np.mean(ys, axis=1)
        dx = xs - cx[:, None]
        dy = ys - cy[:, None]
        out["centroid_x"][rows] = cx
        out["centroid_y"][rows] = cy
        out["team_length"][rows] = np.max(xs, axis=1) - np.min(xs, axis=1)
        out["team_width"][rows] = np.max(ys, axis=1) - np.min(ys, axis=1)
        out["stretch_index"][rows] = np.mean(np.sqrt(dx**2 + dy**2), axis=1)
        out["stretch_x"][rows] = np.mean(np.abs(dx), axis=1)
        out["stretch_y"][rows] = np.mean(np.abs(dy), axis=1)
        if n >= 2:
            out["spread"][rows] = np.sqrt(n * np.sum(dx**2 + dy**2, axis=1))
        if n >= 3:
            out["convex_hull_area"][rows] = _hull_area_fixed_n(pos[rows, :n, :])
    return out


def _select_n_batch(xs_sorted: np.ndarray, n: int | Literal["adaptive"], adaptive_max_n: int, p: int) -> np.ndarray:
    r = xs_sorted.shape[0]
    if n != "adaptive":
        return np.full(r, min(int(n), p), dtype=np.int64)
    if p in (3, 4):
        return np.full(r, p, dtype=np.int64)
    gaps = np.diff(xs_sorted, axis=1)
    cand = [c for c in (3, 4, 5) if (c - 1) < p - 1 and c <= adaptive_max_n]
    default = min(4, p)
    if not cand:
        return np.full(r, default, dtype=np.int64)
    cut = np.abs(gaps[:, [c - 1 for c in cand]])
    mx = cut.max(axis=1)
    second = (-np.sort(-cut, axis=1))[:, 1] if cut.shape[1] > 1 else np.zeros(r)
    pick = np.asarray(cand)[np.argmax(cut, axis=1)]  # first occurrence == list.index(max)
    dominant = (second == 0.0) | (mx >= 1.5 * second)
    return np.where(mx == 0.0, default, np.where(dominant, pick, default)).astype(np.int64)


def back_line_batch(
    pos: np.ndarray,
    counts: np.ndarray,
    defends_x0: np.ndarray,
    *,
    n: int | Literal["adaptive"],
    adaptive_max_n: int,
) -> dict[str, np.ndarray]:
    """Per-row back-line geometry (keys == :data:`BACK_LINE_VARIABLES` + ``"valid"``). ``back_n_count`` is
    int64 (0 where invalid); ``valid`` is ``counts >= 3``. Byte-identical to the TF-14 loop (C1).

    Examples
    --------
    >>> import numpy as np
    >>> pos = np.array([[[10.0, 0.0], [20.0, 5.0], [30.0, 2.0]]])
    >>> bl = back_line_batch(pos, np.array([3]), np.array([True]), n=3, adaptive_max_n=5)
    >>> float(bl["defensive_line_x"][0]), int(bl["back_n_count"][0])
    (20.0, 3)
    """
    g = counts.shape[0]
    out: dict[str, np.ndarray] = {k: np.full(g, np.nan) for k in BACK_LINE_VARIABLES[:-1]}
    back_n = np.zeros(g, dtype=np.int64)
    valid = counts >= 3
    for p in np.unique(counts[valid]):
        rows = np.flatnonzero(counts == p)
        xs, ys, d0 = pos[rows, :p, 0], pos[rows, :p, 1], defends_x0[rows]
        order = np.empty(xs.shape, dtype=np.intp)
        if d0.any():
            order[d0] = np.argsort(xs[d0], axis=1)  # DEFAULT kind == legacy np.argsort(xs)   (C1)
        if (~d0).any():
            order[~d0] = np.argsort(-xs[~d0], axis=1)  # legacy np.argsort(-xs) -- NOT a reversed ascending sort
        xs_s = np.take_along_axis(xs, order, axis=1)
        ys_s = np.take_along_axis(ys, order, axis=1)
        n_eff = _select_n_batch(xs_s, n, adaptive_max_n, int(p))
        for k in np.unique(n_eff):
            sub = np.flatnonzero(n_eff == k)
            sx, sy, r = xs_s[sub, :k], ys_s[sub, :k], rows[sub]
            out["defensive_line_x"][r] = np.mean(sx, axis=1)
            out["compactness_x"][r] = np.max(sx, axis=1) - np.min(sx, axis=1)
            out["back_line_high_x"][r] = np.where(d0[sub], np.max(sx, axis=1), np.min(sx, axis=1))
            out["lateral_width"][r] = np.max(sy, axis=1) - np.min(sy, axis=1)
            out["max_lateral_gap"][r] = np.max(np.diff(np.sort(sy, axis=1), axis=1), axis=1)
            back_n[r] = k
    return {**out, "back_n_count": back_n, "valid": valid}


_CV_REQUIRED = ("game_id", "period_id", "frame_id", "team_id", "is_ball", "is_goalkeeper", "x", "y")


def compute_collective_variables(frames: pd.DataFrame, *, include_goalkeeper: bool = False) -> pd.DataFrame:
    """Per-(game, period, frame, team) collective variables for every team, from tracking frames.

    Outfield players unless ``include_goalkeeper``. One row per group with at least one valid player.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking._collective import compute_collective_variables
    >>> frames = pd.DataFrame(
    ...     {
    ...         "game_id": [1] * 6,
    ...         "period_id": [1] * 6,
    ...         "frame_id": [1, 1, 1, 2, 2, 2],
    ...         "team_id": [7] * 6,
    ...         "is_ball": [False] * 6,
    ...         "is_goalkeeper": [False] * 6,
    ...         "x": [0.0, 4.0, 2.0, 0.0, 4.0, 2.0],
    ...         "y": [0.0, 0.0, 3.0, 0.0, 0.0, 6.0],
    ...     }
    ... )
    >>> out = compute_collective_variables(frames)
    >>> [round(float(v), 3) for v in out["centroid_x"]]
    [2.0, 2.0]
    >>> round(float(out["spread"].iloc[0]), 3)
    6.481
    """
    missing = [c for c in _CV_REQUIRED if c not in frames.columns]
    if missing:
        raise ValueError(f"compute_collective_variables: frames missing columns {missing}")
    cols = ["game_id", "period_id", "frame_id", "team_id", "n_players", *COLLECTIVE_VARIABLES]
    mask = (~frames["is_ball"].astype(bool)) & frames["x"].notna() & frames["y"].notna()
    if not include_goalkeeper:
        mask = mask & (~frames["is_goalkeeper"].astype(bool))
    outfield = frames[mask]
    if outfield.empty:
        return pd.DataFrame(columns=cols)
    gb = outfield.groupby(["game_id", "period_id", "frame_id", "team_id"], dropna=False, sort=True, observed=True)
    codes = gb.ngroup().to_numpy()
    key_tuples = gb.size().index.tolist()
    pos, counts, _first = pack_groups(
        codes, outfield["x"].to_numpy(dtype="float64"), outfield["y"].to_numpy(dtype="float64"), len(key_tuples)
    )
    cv = collective_from_positions(pos, counts)
    result = pd.DataFrame(key_tuples, columns=["game_id", "period_id", "frame_id", "team_id"])
    result["n_players"] = pd.array(counts, dtype="Int64")
    for name in COLLECTIVE_VARIABLES:
        result[name] = cv[name]
    for idc in ("game_id", "team_id"):
        result[idc] = restore_id_dtype(result[idc], frames[idc].dtype)
    return result[cols]
