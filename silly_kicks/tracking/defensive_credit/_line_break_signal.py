"""Per-action between-lines line-break signal for rule_failed_marking_through_ball (TF-51 v2).

Computed home_team_id-free (P-2) in action-LTR coordinates so the shared TF-32 ``_straddle_core``
(the SINGLE straddle implementation) gives the same answer ``detect_line_breaking`` would. The
signal is a firing-condition input, not rule logic, so it is precomputed ONCE per batch on the
orchestrator (spec section 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking._line_breaking import LineBreakingParams, _straddle_core

from ._params import _FIELD_LENGTH, _FIELD_WIDTH

_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]


def precompute_line_break_between_lines(
    act: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    fid_by_pos: np.ndarray,
    flip_by_pos: np.ndarray,
    lb_params: LineBreakingParams | None = None,
) -> pd.arrays.BooleanArray:
    """Per-action nullable-boolean ``between_lines`` line-break signal.

    Candidate = a SUCCESSFUL pass with a linked frame (``rule_failed_marking_through_ball`` fires
    only on those, P10 -- so no Ward clustering runs for any other action). Every non-candidate
    (not a successful pass, unlinked, NaN-actor, no opponents-of-record) -> ``pd.NA``. A candidate
    that reaches the geometry -> ``True`` iff the pass straddles two adjacent same-line defenders
    (``_straddle_core`` returns ``between_lines``); the min_pass_length / min_opponents /
    min_x_spread / <2-cluster short-circuits -> ``False``.

    ``fid_by_pos`` / ``flip_by_pos`` are the orchestrator's once-computed per-action linked frame
    id (NaN when unlinked) and action-LTR reprojection decision (``acting_team_attacks_rtl``).
    """
    lb_params = lb_params or LineBreakingParams()
    n = len(act)
    out = np.array([pd.NA] * n, dtype=object)
    if n == 0 or len(frames) == 0:
        return pd.array(out, dtype="boolean")  # type: ignore[return-value]

    type_id = act["type_id"].to_numpy()
    result_id = act["result_id"].to_numpy()
    sx = act["start_x"].to_numpy(dtype="float64")
    sy = act["start_y"].to_numpy(dtype="float64")
    ex = act["end_x"].to_numpy(dtype="float64")
    ey = act["end_y"].to_numpy(dtype="float64")
    team_ids = act["team_id"].to_numpy()

    fr_by_frame = dict(iter(frames.groupby("frame_id", sort=False)))

    for i in range(n):
        if type_id[i] != _PASS or result_id[i] != _SUCCESS:
            continue  # non-candidate -> pd.NA (no Ward clustering, P10)
        fid = fid_by_pos[i]
        if pd.isna(fid):
            continue  # unlinked -> pd.NA
        fr = fr_by_frame.get(int(fid))
        if fr is None:
            continue
        acting = team_ids[i]
        if pd.isna(acting):
            continue  # ADR-027: NaN-actor never decides
        # opponent outfielders (team != acting, non-ball, non-GK -- mirrors _line_breaking:152)
        is_opp = (
            ~ids_match(fr["team_id"], acting)
            & fr["team_id"].notna()
            & ~fr["is_ball"].astype(bool)
            & ~fr["is_goalkeeper"].astype(bool)
        )
        opp = fr[is_opp.to_numpy()]
        if opp.empty:
            continue  # no opponents of record -> pd.NA (indistinguishable from unlinked)
        ox = opp["x"].to_numpy(dtype="float64")
        oy = opp["y"].to_numpy(dtype="float64")
        valid = ~np.isnan(ox) & ~np.isnan(oy)
        ox, oy = ox[valid], oy[valid]
        # reproject opponents to action-LTR (the family's scalar-flip idiom, _resolution.py:53-54)
        if bool(flip_by_pos[i]):
            ox = _FIELD_LENGTH - ox
            oy = _FIELD_WIDTH - oy
        is_break, break_type, _n = _straddle_core(sx[i], sy[i], ex[i], ey[i], ox, oy, lb_params)
        out[i] = bool(is_break and break_type == "between_lines")

    return pd.array(out, dtype="boolean")  # type: ignore[return-value]
