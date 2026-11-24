"""Shared corpus helpers for the cover-shadow RQ1 + pass-risk validation cycle.

Played-pass extraction + the ADR-028 orientation helper. Consumed by ``build_rq_pass_scores``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import ids_match
from silly_kicks.spadl import config as spc
from silly_kicks.spadl.utils import resolve_next_touch_receiver
from silly_kicks.tracking import link_actions_to_frames

_PASS_TYPES = {spc.actiontype_id[t] for t in ("pass", "cross")}  # spec 4: pass/cross ONLY
_CROSS = spc.actiontype_id["cross"]  # crosses are aerial -> Driver A headline is pass-only
_SUCCESS = spc.result_id["success"]


def to_frame_coords(x: float, y: float, attacks_rtl: bool) -> tuple[float, float]:
    """Action-LTR (acting team attacks x=105) -> frame convention (home-attacks-right).

    Away-team actions are a 180-degree point reflection (ADR-028); home actions are already aligned.
    """
    return (105.0 - x, 68.0 - y) if attacks_rtl else (x, y)


def _acting_attacks_rtl(fr: pd.DataFrame, team_id) -> bool:
    """GS frames are home-attacks-right; a team's own player rows carry ``team_attacking_direction``.

    A frame that does not DECLARE orientation (no such column -- SB360 snapshots, synthetic fixtures) is
    treated as aligned (action-LTR == frame) -> ``False``, no reflection. ``is_ball`` via
    ``to_numpy(dtype=bool)`` NOT ``.astype(bool)`` on a possibly-object column (ADR-019).
    """
    if "team_attacking_direction" not in fr.columns:
        return False
    prow = fr[ids_match(fr["team_id"], team_id) & ~fr["is_ball"].to_numpy(dtype=bool)]
    return (not prow.empty) and str(prow["team_attacking_direction"].iloc[0]) == "rtl"


def _player_frame_xy(fr: pd.DataFrame, pid) -> tuple[float, float] | None:
    row = fr[ids_match(fr["player_id"], pid)]
    return None if row.empty else (float(row["x"].iloc[0]), float(row["y"].iloc[0]))


def extract_played_passes(
    actions: pd.DataFrame, frames: pd.DataFrame, *, links: pd.DataFrame | None = None, model=None
) -> pd.DataFrame:
    """One row per played pass: passer + target (frame coords) + outcome, for the score driver.

    Completed pass -> the observed receiver's frame position (leakage-free). Failed pass: with
    ``model=None`` the release-frame ``end_xy`` proxy (outcome-selected -- the 4.87.0 leaked behaviour,
    kept byte-identical); with a receiver ``model`` the DE-LEAKED, failure-mode-conditional target
    (Task 8) -- intercepted -> the model's intended receiver, out -> the overshot (non-defender-selected)
    end direction, other -> the ``end_xy`` fallback. ``target_source`` records which.
    """
    from silly_kicks.tracking._link_identity import linked_frame_rows, pointers_by_action_row

    is_pass = actions["type_id"].isin(_PASS_TYPES).to_numpy()
    passes = actions[is_pass].copy()
    if links is None:  # link_actions_to_frames returns (pointers, LinkReport) -- ADR-004
        links, _ = link_actions_to_frames(actions, frames)
    # Each pass reads its pointer and its OWN game's frame BY POSITION (spec 2026-11-08 tracking-batch
    # section 4.9 W9.4, R8): an action_id map or a frame_id-only lookup crosses games.
    pass_rows = np.flatnonzero(is_pass)
    passes["frame_id"] = pointers_by_action_row(actions, links)["frame_id"].to_numpy(dtype=object)[pass_rows]
    passes["_row"] = pass_rows
    passes = passes[passes["frame_id"].notna()]
    receiver_id = resolve_next_touch_receiver(actions).reindex(passes.index)
    fmode_by_row = None
    if model is not None:  # de-leaked path needs the failure-mode split (interception vs out)
        from scripts._receiver_validation import _failure_modes_by_row

        fmode_by_row = _failure_modes_by_row(actions)
    frames_of = linked_frame_rows(actions, frames, links)  # index ONCE per match
    rows = []
    for idx, a in passes.iterrows():
        row = int(a["_row"])
        fr = frames_of[row]
        if fr.empty:
            continue
        fmode = None if fmode_by_row is None else fmode_by_row[row]
        attacks_rtl = _acting_attacks_rtl(fr, a["team_id"])
        is_completed = int(a["result_id"]) == _SUCCESS
        rid = receiver_id.get(idx)
        rec_xy = _player_frame_xy(fr, rid) if (is_completed and pd.notna(rid)) else None
        if rec_xy is not None:  # receiver read from frame -> already frame coords, NOT reflected
            tx, ty, src = rec_xy[0], rec_xy[1], "receiver"
        elif (not is_completed) and model is not None:  # DE-LEAKED failed-pass target (Task 8, M2)
            intended_xy = None
            if fmode == "intercepted":
                ranked = model.rank(a, fr)  # model target is a frame position, NOT reflected
                if not ranked.empty:
                    intended_xy = _player_frame_xy(fr, ranked.index[0])
            if intended_xy is not None:
                tx, ty, src = intended_xy[0], intended_xy[1], "intended_receiver"
            else:  # out (overshot, not defender-selected) / other -> end direction, tagged by mode
                tx, ty = to_frame_coords(float(a["end_x"]), float(a["end_y"]), attacks_rtl)
                src = "trajectory" if fmode == "out" else "end_xy_legacy"
        else:  # model=None -> 4.87.0 leaked behaviour, byte-identical. end_xy is action-LTR -> reflect for away
            tx, ty = to_frame_coords(float(a["end_x"]), float(a["end_y"]), attacks_rtl)
            src = "end_xy"
        px, py = to_frame_coords(float(a["start_x"]), float(a["start_y"]), attacks_rtl)
        rows.append(
            {
                "game_id": a["game_id"],
                "period_id": a["period_id"],
                "action_id": a["action_id"],
                "frame_id": a["frame_id"],
                "attacking_team_id": a["team_id"],
                "passer_x": px,
                "passer_y": py,
                "target_x": tx,
                "target_y": ty,
                "target_source": src,
                "is_cross": int(a["type_id"]) == _CROSS,
                "is_completed": is_completed,
                "is_fail": not is_completed,
            }
        )
    return pd.DataFrame(rows)
