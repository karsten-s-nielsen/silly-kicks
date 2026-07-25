"""Pressure-commitment cue (TF-51 v2 Item 5, spec section 6).

A per-action DESCRIPTOR of whether the defender pressing the actor COMMITS (drives in) versus
CONTAINS (decelerates to jockey). Role (A): a press style/quality descriptor, NOT signed credit --
it composes with the defensive-credit rules but is not itself a value, so it lives here (outside
``defensive_credit/``) and ships aggregator-only (no ``*_xfns``).

Metric: axis = the unit defender->actor vector FIXED at the action frame; closing-speed at a window
frame = that frame's ``(vx, vy) . axis``; commitment = the LEAST-SQUARES SLOPE of closing-speed over
the window (m/s^2; positive = committing, negative = braking) on a fixed >=0.1 s baseline (no
sub-baseline fallback). The defender->actor axis + projection are a RELATIVE vector between two
players in one frame -- direction-agnostic, so no ADR-028 reprojection is needed for the scalar.

See NOTICE for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match

from ._opponent_resolution import opponents_action_ltr
from ._velocity_availability import velocity_unavailable_by_design

# --- closed source vocabulary (DAS_SOURCE_VALUES pattern, spec section 6) ---
PRESS_COMMITMENT_COMPUTED = "computed"
PRESS_COMMITMENT_NO_PRESSING_DEFENDER = "no_pressing_defender"
PRESS_COMMITMENT_VELOCITY_UNAVAILABLE = "velocity_unavailable"
PRESS_COMMITMENT_WINDOW_TOO_SHORT = "window_too_short"
PRESS_COMMITMENT_DEGENERATE_AXIS = "degenerate_axis"
PRESS_COMMITMENT_UNLINKED = "unlinked"
PRESS_COMMITMENT_SOURCE_VALUES: tuple[str, ...] = (
    PRESS_COMMITMENT_COMPUTED,
    PRESS_COMMITMENT_NO_PRESSING_DEFENDER,
    PRESS_COMMITMENT_VELOCITY_UNAVAILABLE,
    PRESS_COMMITMENT_WINDOW_TOO_SHORT,
    PRESS_COMMITMENT_DEGENERATE_AXIS,
    PRESS_COMMITMENT_UNLINKED,
)

_MIN_BASELINE_SECONDS = 0.1
_OUTPUT_COLS = ("press_commitment", "press_commitment_closing_speed", "press_commitment_source")


@dataclass(frozen=True)
class PressCommitmentParams:
    """All fields intent-set / NEVER calibrated (provisional starting values, spec section 6/11)."""

    commitment_window_seconds: float = 0.5  # the run-up over which closing-speed is fit
    press_max_distance_m: float = 3.0  # a defender beyond this is not pressing -> no_pressing_defender
    min_separation_m: float = 0.5  # below this the defender->actor axis is ill-conditioned

    def __post_init__(self) -> None:
        for name, val in (
            ("commitment_window_seconds", self.commitment_window_seconds),
            ("press_max_distance_m", self.press_max_distance_m),
            ("min_separation_m", self.min_separation_m),
        ):
            if not val > 0:
                raise ValueError(f"{name} must be > 0, got {val}")


def _least_squares_slope(t: np.ndarray, v: np.ndarray) -> float:
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(tc, v - v.mean()) / denom)


def compute_press_commitment(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
    params: PressCommitmentParams | None = None,
) -> pd.DataFrame:
    """Per-action press-commitment cue, aligned to ``actions.index``.

    Columns: ``press_commitment`` (float, m/s^2; + committing / - containing),
    ``press_commitment_closing_speed`` (float, m/s, context), ``press_commitment_source``.

    Reads off ``actions`` only ids + ``time_seconds`` (linking) + ``team_id`` / ``player_id`` (to
    resolve the actor's frame row) -- NOT ``start_x``/``start_y`` (the actor and the pressing defender
    both come from the linked FRAME). Velocity contract: ``speed_source`` all-``unavailable`` ->
    NaN + ``velocity_unavailable``; ``vx``/``vy`` absent and NOT declared unavailable -> loud raise.
    """
    params = params or PressCommitmentParams()
    n = len(actions)
    pc = np.full(n, np.nan)
    cs = np.full(n, np.nan)
    src = np.array([PRESS_COMMITMENT_UNLINKED] * n, dtype=object)
    if n == 0 or len(frames) == 0:
        return _result(actions, pc, cs, src)

    if velocity_unavailable_by_design(frames):
        src[:] = PRESS_COMMITMENT_VELOCITY_UNAVAILABLE
        return _result(actions, pc, cs, src)
    if "vx" not in frames.columns or "vy" not in frames.columns:
        raise ValueError(
            "compute_press_commitment requires vx/vy on frames (call derive_velocities() first), or "
            "declare speed_source unavailable. See the velocity-availability contract."
        )

    from .utils import link_actions_to_frames  # function-local: keep the utils import lazy

    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
    act = actions.reset_index(drop=True)
    fid_by_pos = (
        pointers.drop_duplicates("action_id")
        .set_index("action_id")["frame_id"]
        .reindex(act["action_id"].to_numpy())
        .to_numpy()
    )

    frames_by_frame = dict(iter(frames.groupby("frame_id", sort=False)))
    frames_by_player = dict(iter(frames.groupby("player_id", sort=False)))

    w = params.commitment_window_seconds
    for i in range(n):
        a = act.iloc[i]
        fid = fid_by_pos[i]
        if pd.isna(fid):
            continue  # unlinked
        fr = frames_by_frame.get(int(fid))
        if fr is None:
            continue  # unlinked

        actor = fr[ids_match(fr["player_id"], a["player_id"]).to_numpy() & ~fr["is_ball"].astype(bool).to_numpy()]
        if actor.empty:
            continue  # actor not tracked in the linked frame -> unlinked (no usable actor)
        ax = float(actor["x"].iloc[0])
        ay = float(actor["y"].iloc[0])

        # nearest opponent within press_max_distance (frame coords; a relative vector is flip-invariant)
        opp = opponents_action_ltr(fr, a["team_id"], flip=False, exclude_goalkeeper=False)  # a keeper may press
        if opp.empty:
            src[i] = PRESS_COMMITMENT_NO_PRESSING_DEFENDER
            continue
        odx = opp["_px"].to_numpy() - ax
        ody = opp["_py"].to_numpy() - ay
        odist = np.hypot(odx, ody)
        within = odist <= params.press_max_distance_m
        if not within.any():
            src[i] = PRESS_COMMITMENT_NO_PRESSING_DEFENDER
            continue
        cand = np.where(within)[0]
        j = int(cand[np.argmin(odist[cand])])
        def_id = opp["player_id"].to_numpy()[j]
        sep = float(odist[j])
        if sep < params.min_separation_m:
            src[i] = PRESS_COMMITMENT_DEGENERATE_AXIS
            continue
        axis = np.array([ax - float(opp["_px"].to_numpy()[j]), ay - float(opp["_py"].to_numpy()[j])]) / sep

        g = frames_by_player.get(def_id)
        if g is None:
            src[i] = PRESS_COMMITMENT_WINDOW_TOO_SHORT
            continue
        game_id = fr["game_id"].iloc[0]
        period_id = fr["period_id"].iloc[0]
        t0 = float(a["time_seconds"])
        wmask = (
            (g["game_id"] == game_id)
            & (g["period_id"] == period_id)
            & (g["time_seconds"] >= t0 - w)
            & (g["time_seconds"] <= t0)
        )
        win = g[wmask.to_numpy()]
        wt = win["time_seconds"].to_numpy(dtype="float64")
        vcx = win["vx"].to_numpy(dtype="float64")
        vcy = win["vy"].to_numpy(dtype="float64")
        v_close = vcx * axis[0] + vcy * axis[1]
        good = np.isfinite(v_close) & np.isfinite(wt)
        if good.sum() < 2 or float(wt[good].max() - wt[good].min()) < _MIN_BASELINE_SECONDS:
            src[i] = PRESS_COMMITMENT_WINDOW_TOO_SHORT  # no sub-baseline fallback (spec section 6)
            continue

        pc[i] = _least_squares_slope(wt[good], v_close[good])
        cs[i] = float(v_close[good][np.argmax(wt[good])])  # closing speed at the frame nearest the action
        src[i] = PRESS_COMMITMENT_COMPUTED

    return _result(actions, pc, cs, src)


def _result(actions: pd.DataFrame, pc: np.ndarray, cs: np.ndarray, src: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "press_commitment": pc,
            "press_commitment_closing_speed": cs,
            "press_commitment_source": pd.array(src, dtype="object"),
        },
        index=actions.index,
    )
