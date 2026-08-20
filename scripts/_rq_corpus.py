"""Shared corpus helpers for the cover-shadow RQ1 + pass-risk validation cycle.

Played-pass extraction + the ADR-028 orientation helper. Consumed by ``build_rq_pass_scores``.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import canonical_id, ids_match
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

    ``is_ball`` via ``to_numpy(dtype=bool)`` NOT ``.astype(bool)`` on a possibly-object column (ADR-019).
    """
    prow = fr[ids_match(fr["team_id"], team_id) & ~fr["is_ball"].to_numpy(dtype=bool)]
    return (not prow.empty) and str(prow["team_attacking_direction"].iloc[0]) == "rtl"


def _player_frame_xy(fr: pd.DataFrame, pid) -> tuple[float, float] | None:
    row = fr[ids_match(fr["player_id"], pid)]
    return None if row.empty else (float(row["x"].iloc[0]), float(row["y"].iloc[0]))


def extract_played_passes(
    actions: pd.DataFrame, frames: pd.DataFrame, *, links: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per played pass: passer + target (frame coords) + outcome, for the score driver.

    Completed pass -> the observed receiver's frame position (leakage-free); failed pass -> the
    release-frame ``end_xy`` proxy (outcome-selected -> see the spec's leakage caveat).
    """
    passes = actions[actions["type_id"].isin(_PASS_TYPES)].copy()
    if links is None:  # link_actions_to_frames returns (pointers, LinkReport) -- ADR-004
        links, _ = link_actions_to_frames(actions, frames)
    passes["frame_id"] = passes["action_id"].map(links.set_index("action_id")["frame_id"])
    passes = passes[passes["frame_id"].notna()]
    receiver_id = resolve_next_touch_receiver(actions).reindex(passes.index)
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}  # index ONCE per match
    rows = []
    for idx, a in passes.iterrows():
        fr = by_frame.get(canonical_id(a["frame_id"]))
        if fr is None:
            continue
        attacks_rtl = _acting_attacks_rtl(fr, a["team_id"])
        is_completed = int(a["result_id"]) == _SUCCESS
        rid = receiver_id.get(idx)
        rec_xy = _player_frame_xy(fr, rid) if (is_completed and pd.notna(rid)) else None
        if rec_xy is not None:  # receiver read from frame -> already frame coords, NOT reflected
            tx, ty, src = rec_xy[0], rec_xy[1], "receiver"
        else:  # end_xy is action-LTR -> reflect for away
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
