"""Progression / possession KPI family (TF-52 Sections 4.2-4.3).

Per ``(game_id, team_id)``: field tilt, pass tempo, long-ball %, the three line heights, the
conversion chain, final-third entries + shots (raw counts), high-opportunity shots (injected xG),
breakout-by-channel, and possessions-retained-after-Ns. Own-touch KPIs use each team's own
action-LTR frame (no reflection);
field tilt is a share of both teams' own-frame final-third open-play touches. Honest-NaN on empty
denominators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id, ids_match
from silly_kicks.spadl import config as spadlconfig

from ._possession import _SET_PIECE_TYPE_IDS

_FL = spadlconfig.field_length
_FW = spadlconfig.field_width
_PASS = spadlconfig.actiontype_id["pass"]
_SHOT_TYPE_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_NON_PENALTY_SHOT_TYPE_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_freekick")]

_PROGRESSION_COLUMNS = [
    "game_id",
    "team_id",
    "field_tilt_pct",
    "pass_tempo",
    "long_ball_pct",
    "defensive_action_height_m",
    "recovery_line_height_m",
    "turnover_line_height_m",
    "poss_to_final_third_pct",
    "final_third_entries",
    "final_third_to_box_pct",
    "box_touches",
    "box_to_shot_pct",
    "shots",
    "high_opportunity_shots",
    "breakout_left",
    "breakout_center",
    "breakout_right",
    "breakout_left_pct",
    "breakout_center_pct",
    "breakout_right_pct",
    "possessions_retained_after_ns_pct",
]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else np.nan


def compute_progression_kpis(
    actions_ctx: pd.DataFrame, spells: pd.DataFrame, minutes: pd.DataFrame, *, xg_column, params
) -> pd.DataFrame:
    """One row per ``(game_id, team_id)`` with the progression KPIs (see module docstring)."""
    a = actions_ctx
    def_ids = [spadlconfig.actiontype_id[n] for n in params.defensive_action_types]
    box_min_x = _FL - spadlconfig.penalty_area_depth
    box_half_w = spadlconfig.penalty_area_half_width
    ft_min_x = 2.0 / 3.0 * _FL
    half_x = _FL / 2.0
    c_lo, c_hi = params.channel_boundaries
    has_xg = xg_column is not None and xg_column in a.columns

    in_box = (a["start_x"] >= box_min_x) & ((a["start_y"] - _FW / 2.0).abs() <= box_half_w)
    in_ft = a["start_x"] >= ft_min_x
    is_open_play = ~a["type_id"].isin(_SET_PIECE_TYPE_IDS)

    # per-possession features (reached final third / box / had a shot), joined onto spells.
    feat = (
        a.assign(_ft=in_ft, _box=in_box, _shot=a["type_id"].isin(_SHOT_TYPE_IDS))
        .groupby(["game_id", "possession_id"], sort=False)
        .agg(reached_ft=("_ft", "max"), reached_box=("_box", "max"), had_shot=("_shot", "max"))
        .reset_index()
    )
    crossed = a[a["start_x"] >= half_x]
    cross_y = (
        crossed.groupby(["game_id", "possession_id"], sort=False)["start_y"].first().rename("cross_y").reset_index()
    )
    sf = spells.merge(feat, on=["game_id", "possession_id"], how="left").merge(
        cross_y, on=["game_id", "possession_id"], how="left"
    )
    sf["is_breakout"] = sf["is_open_play"] & (sf["start_x_ltr"] < half_x) & sf["cross_y"].notna()
    sf["channel"] = np.where(sf["cross_y"] < c_lo, "left", np.where(sf["cross_y"] >= c_hi, "right", "center"))

    # game-total final-third open-play touches (denominator of field tilt) -> canonical-keyed dict.
    op_ft = a[is_open_play & in_ft].dropna(subset=["team_id"])
    game_total_ft = {canonical_id(g): int(n) for g, n in op_ft.groupby("game_id", sort=False).size().items()}

    game_groups = group_rows(a, "game_id")
    spell_groups = group_rows(sf, ("game_id", "team_id"))
    minute_groups = group_rows(minutes, ("game_id", "team_id"))

    rows: list[dict] = []
    for game_id in pd.unique(a["game_id"]):
        ga = game_groups.get(game_id)
        total_ft = game_total_ft.get(canonical_id(game_id), 0)
        for team_id in pd.unique(ga["team_id"].dropna()):
            team_actions = ga.iloc[ids_match(ga["team_id"], team_id).to_numpy()]
            team_spells = spell_groups.get(game_id, team_id)
            mrow = minute_groups.get(game_id, team_id)
            in_min = float(mrow["in_possession_min"].iloc[0]) if len(mrow) else np.nan
            rows.append(
                _progression_row(
                    game_id,
                    team_id,
                    team_actions,
                    team_spells,
                    in_min,
                    total_ft,
                    def_ids,
                    box_min_x,
                    box_half_w,
                    ft_min_x,
                    half_x,
                    has_xg,
                    xg_column,
                    params,
                )
            )
    return pd.DataFrame(rows, columns=_PROGRESSION_COLUMNS)


def _progression_row(
    game_id,
    team_id,
    team_actions,
    team_spells,
    in_min,
    total_ft,
    def_ids,
    box_min_x,
    box_half_w,
    ft_min_x,
    half_x,
    has_xg,
    xg_column,
    params,
) -> dict:
    ta = team_actions
    row: dict = {"game_id": game_id, "team_id": team_id}

    tx = ta["start_x"].to_numpy(dtype="float64")
    ty = ta["start_y"].to_numpy(dtype="float64")
    ttype = ta["type_id"].to_numpy()
    is_op = ~np.isin(ttype, list(_SET_PIECE_TYPE_IDS))

    # field tilt = own final-third open-play touches / both teams' final-third open-play touches.
    own_ft = int(np.count_nonzero(is_op & (tx >= ft_min_x)))
    row["field_tilt_pct"] = _safe_div(own_ft, total_ft)

    # pass tempo = passes per minute of possession.
    n_pass = int(np.count_nonzero(ttype == _PASS))
    row["pass_tempo"] = (n_pass / in_min) if (in_min is not None and in_min > 0) else np.nan

    # long-ball % = own-half passes > distance / own-half passes.
    is_pass = ttype == _PASS
    own_half = tx < half_x
    ex = ta["end_x"].to_numpy(dtype="float64")
    ey = ta["end_y"].to_numpy(dtype="float64")
    plen = np.hypot(ex - tx, ey - ty)
    own_half_pass = is_pass & own_half
    n_own_half_pass = int(np.count_nonzero(own_half_pass))
    n_long = int(np.count_nonzero(own_half_pass & (plen > params.long_ball_distance_m)))
    row["long_ball_pct"] = _safe_div(n_long, n_own_half_pass)

    # line heights.
    is_def = np.isin(ttype, def_ids)
    row["defensive_action_height_m"] = float(np.mean(tx[is_def])) if is_def.any() else np.nan
    op = team_spells[team_spells["is_open_play"]]
    rec = op[op["is_recovery"]]
    row["recovery_line_height_m"] = float(rec["start_x_ltr"].mean()) if len(rec) else np.nan
    row["turnover_line_height_m"] = float(op["last_x_ltr"].mean()) if len(op) else np.nan

    # conversion chain over open-play possessions.
    n_poss = len(op)
    reached_ft = int(op["reached_ft"].fillna(False).sum())
    reached_box = int((op["reached_ft"].fillna(False) & op["reached_box"].fillna(False)).sum())
    box_and_shot = int((op["reached_box"].fillna(False) & op["had_shot"].fillna(False)).sum())
    n_box_reaching = int(op["reached_box"].fillna(False).sum())
    row["poss_to_final_third_pct"] = _safe_div(reached_ft, n_poss)
    row["final_third_to_box_pct"] = _safe_div(reached_box, reached_ft)
    row["box_to_shot_pct"] = _safe_div(box_and_shot, n_box_reaching)

    # final-third entries: per-action count of the ball CROSSING x = 2*FL/3 into the final third
    # (start outside, end inside), in the team's own attacking frame. Raw count, xG-independent --
    # the complementary count to the possession-rate poss_to_final_third_pct.
    entered_ft = (tx < ft_min_x) & (ex >= ft_min_x)
    row["final_third_entries"] = int(np.count_nonzero(entered_ft))

    # box touches (count of actions started in the attacking box).
    in_box = (tx >= box_min_x) & (np.abs(ty - _FW / 2.0) <= box_half_w)
    row["box_touches"] = int(np.count_nonzero(in_box))

    # shots: raw count of all shots (open-play + set-piece + penalty) by the team. xG-independent.
    row["shots"] = int(np.count_nonzero(np.isin(ttype, _SHOT_TYPE_IDS)))

    # high-opportunity shots (non-penalty shots with injected xG over threshold; NaN if no xG).
    if has_xg:
        xg = ta[xg_column].to_numpy(dtype="float64")
        is_np_shot = np.isin(ttype, _NON_PENALTY_SHOT_TYPE_IDS)
        row["high_opportunity_shots"] = int(np.count_nonzero(is_np_shot & (xg > params.high_opportunity_xg)))
    else:
        row["high_opportunity_shots"] = np.nan

    # breakouts by channel (share of breakouts per channel).
    bo = op[op["is_breakout"]]
    counts = {"left": 0, "center": 0, "right": 0}
    for ch, n in bo["channel"].value_counts().items():
        counts[ch] = int(n)
    total_bo = sum(counts.values())
    for ch in ("left", "center", "right"):
        row[f"breakout_{ch}"] = counts[ch]
        row[f"breakout_{ch}_pct"] = _safe_div(counts[ch], total_bo)

    # possessions retained after N s (open-play possessions held >= threshold).
    if n_poss > 0:
        retained = int((op["duration_s"] >= params.retained_after_seconds).sum())
        row["possessions_retained_after_ns_pct"] = float(retained / n_poss)
    else:
        row["possessions_retained_after_ns_pct"] = np.nan

    return row
