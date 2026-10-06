"""Broadcast-occlusion simulation for the D1 occlusion pass (spec 8.2).

Fully-observed providers (Sportec/IDSSE/GradientSports) carry every player every frame; SkillCorner broadcast
tracking does not. To measure how coordination metrics degrade under SkillCorner-like partial detection WITHOUT
conflating it with SkillCorner's other quirks, we take a fully-observed match and blank the players outside a
moving broadcast field-of-view window, then extrapolate them exactly as SkillCorner would -- a labelled
APPROXIMATION of the real censoring, not a claim to reproduce it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PITCH_X = 105.0


def fov_mask(frames: pd.DataFrame, *, width_m: float) -> np.ndarray:
    """Per-row broadcast-visibility mask: a player is visible iff its x is inside the frame's FOV window.

    The window is ``[c - W/2, c + W/2]`` with ``c`` the ball's x that frame, clamped so the window stays on the
    pitch (``c in [W/2, 105 - W/2]``). Ball rows are always ``True``.
    """
    x = frames["x"].to_numpy(dtype=np.float64)
    is_ball = frames["is_ball"].to_numpy(dtype=bool)
    half = width_m / 2.0
    ball_x = _ball_x_per_frame(frames)
    centre = np.clip(ball_x, half, _PITCH_X - half)
    inside = (x >= centre - half) & (x <= centre + half)
    return inside | is_ball


def _ball_x_per_frame(frames: pd.DataFrame) -> np.ndarray:
    """The ball's x broadcast to every row of its (game, period, frame); NaN where the frame has no ball row."""
    ball = frames.loc[frames["is_ball"], ["game_id", "period_id", "frame_id", "x"]]
    key_cols = ["game_id", "period_id", "frame_id"]
    ball_x = ball.drop_duplicates(key_cols).set_index(key_cols)["x"]
    idx = pd.MultiIndex.from_arrays([frames[c] for c in key_cols])
    return ball_x.reindex(idx).to_numpy(dtype=np.float64)


def simulate_broadcast_occlusion(frames: pd.DataFrame, *, width_m: float) -> pd.DataFrame:
    """Return a copy with ``visibility`` set to the FOV mask and masked player positions extrapolated.

    Masked (out-of-FOV) player rows keep a position, but it is linearly interpolated between that player's
    detected frames (edges held) -- an approximation of SkillCorner's own extrapolation, labelled as one. Ball
    rows are untouched. Detected rows keep their true position.
    """
    out = frames.copy()
    mask = fov_mask(frames, width_m=width_m)
    out["visibility"] = pd.array(np.where(frames["is_ball"].to_numpy(bool), None, mask), dtype="object")
    cols = {axis: (_column_position(out, axis), out[axis].to_numpy().dtype) for axis in ("x", "y")}
    for _key, sub in player_period_row_groups(out):
        det = mask[sub]
        for axis, (col, dtype) in cols.items():
            vals = out[axis].to_numpy(dtype=np.float64)[sub]
            # cast back to the stored dtype: coords are float32 storage (ADR-106), and pandas 3.0
            # refuses a lossy float64 write into a float32 column (mirrors _interpolation.py:101).
            out.iloc[sub, col] = _interp_masked(vals, det).astype(dtype)
    return out


def _column_position(frames: pd.DataFrame, column: str) -> int:
    """The integer position of ``column``; a duplicated label (``get_loc`` -> slice/mask) is refused."""
    loc = frames.columns.get_loc(column)
    if not isinstance(loc, (int, np.integer)):
        raise ValueError(f"column {column!r} is not unique in the frames")
    return int(loc)


def player_period_row_groups(frames: pd.DataFrame):
    """Yield ``((game_id, period_id, player_id), row positions)`` per player WITHIN each period, rows in time order.

    ``time_seconds`` is period-relative (ADR-017): a player's rows ordered by it across periods interleave the two
    halves sample by sample (review A-04). Ball rows and rows without a player id are skipped; a nullable id column
    (``Int64`` with the ball row's ``<NA>``, ADR-058) is grouped, never compared element by element.
    """
    keep = ~frames["is_ball"].to_numpy(dtype=bool) & frames["player_id"].notna().to_numpy(dtype=bool)
    if not keep.any():
        return
    sub = frames.loc[keep, ["game_id", "period_id", "player_id", "time_seconds"]].assign(_pos=np.flatnonzero(keep))
    for key, g in sub.groupby(["game_id", "period_id", "player_id"], sort=True, observed=True):
        yield key, g.sort_values("time_seconds", kind="mergesort")["_pos"].to_numpy()


def _interp_masked(values: np.ndarray, detected: np.ndarray) -> np.ndarray:
    """Linear-interpolate the ``~detected`` entries of ``values`` from the detected ones (edges held)."""
    out = values.astype(np.float64).copy()
    if detected.all() or not detected.any():
        return out
    i = np.arange(len(out))
    out[~detected] = np.interp(i[~detected], i[detected], out[detected])
    return out


def detection_rates(frames: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]:
    """(outfield, goalkeeper) detected fraction over the player rows under ``mask``."""
    is_ball = frames["is_ball"].to_numpy(bool)
    is_gk = frames["is_goalkeeper"].to_numpy(bool)
    outfield = ~is_ball & ~is_gk
    gk = ~is_ball & is_gk
    of_rate = float(mask[outfield].mean()) if outfield.any() else float("nan")
    gk_rate = float(mask[gk].mean()) if gk.any() else float("nan")
    return of_rate, gk_rate


def calibrate_width(frames_list, *, target_outfield: float = 0.666, tol: float = 0.002) -> float:
    """Bisection on the FOV width ``W in [5, 105]`` that yields ``target_outfield`` outfield detection.

    Detection rises monotonically with ``W`` (a wider window sees more players), so a bisection converges. The
    rate is pooled over ``frames_list`` (each match's outfield rows weighted equally by row count).
    """

    def rate(width: float) -> float:
        num = den = 0.0
        for frames in frames_list:
            m = fov_mask(frames, width_m=width)
            of = (~frames["is_ball"].to_numpy(bool)) & (~frames["is_goalkeeper"].to_numpy(bool))
            num += float(m[of].sum())
            den += float(of.sum())
        return num / den if den else float("nan")

    lo, hi = 5.0, _PITCH_X
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        r = rate(mid)
        if abs(r - target_outfield) <= tol:
            return mid
        if r < target_outfield:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
