"""Build-up & post-regain KPI family (TF-52 Section 4.4).

Per ``(game_id, team_id)``:
- Build-up outcome taxonomy (6 states + success %). A *build-up* is an open-play possession whose
  first action starts in the own build-up zone (``start_x_ltr < build_up_zone_max_x``). Classified
  into exactly one state, evaluated in this documented priority order (first match wins):
  1. ``final_quarter``  -- reached ``x >= 3/4 * field_length``
  2. ``next_phase``     -- crossed ``field_length / 2`` (but not the final quarter)
  3. ``led_opp_shot``   -- lost to the opponent whose next possession produced a shot
  4. ``opp_int_own_half`` -- lost via an opponent interception, turnover in the own half
  5. ``opp_won_own_half`` -- lost in the own half (any other way)
  6. ``stayed_phase_one`` -- default (no progression, no clear own-half loss)
  ``buildup_success_pct = (final_quarter + next_phase) / build-ups``.
- Post-regain security (over recovery possessions): 2nd-pass completion rate, failed-first-pass count,
  forward-first-option %.
- Switch-conditioned press (over the OPPONENT's short goal kicks): ``switch_press_n`` = the count of
  opponent short goal kicks faced (small-N, surfaced), ``switch_press_success_pct`` = the fraction the
  team both kept switch-free and regained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import ids_differ, ids_match
from silly_kicks.spadl import config as spadlconfig

_FL = spadlconfig.field_length
_FW = spadlconfig.field_width
_PASS = spadlconfig.actiontype_id["pass"]
_GOALKICK = spadlconfig.actiontype_id["goalkick"]
_INTERCEPTION = spadlconfig.actiontype_id["interception"]
_SHOT_TYPE_IDS = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
_SUCCESS = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]

_BUILDUP_STATE_COLUMNS = {
    "final_quarter": "buildup_final_quarter",
    "next_phase": "buildup_next_phase",
    "opp_int_own_half": "buildup_opp_int_own_half",
    "stayed_phase_one": "buildup_stayed_phase_one",
    "opp_won_own_half": "buildup_opp_won_own_half",
    "led_opp_shot": "buildup_led_opp_shot",
}

_BUILDUP_COLUMNS = [
    "game_id",
    "team_id",
    "buildup_final_quarter",
    "buildup_next_phase",
    "buildup_opp_int_own_half",
    "buildup_stayed_phase_one",
    "buildup_opp_won_own_half",
    "buildup_led_opp_shot",
    "buildup_success_pct",
    "post_regain_second_pass_pct",
    "post_regain_failed_first_passes",
    "post_regain_forward_first_pct",
    "switch_press_success_pct",
    "switch_press_n",
]


def compute_buildup_kpis(actions_ctx: pd.DataFrame, spells: pd.DataFrame, *, params) -> pd.DataFrame:
    """One row per ``(game_id, team_id)`` with the build-up / post-regain / switch KPIs."""
    a = actions_ctx
    half_x = _FL / 2.0
    fq_x = 0.75 * _FL

    sf = _possession_frame(a, spells, params)
    post = _post_regain_summary(a, sf)
    sgk = sf[sf["is_short_goalkick"]]

    # classify build-ups (vectorized) and tally per (game, team).
    bu = _classify_buildups(sf, params.build_up_zone_max_x, half_x, fq_x)

    # Group ONCE per game (ADR-068: no full-table rescan inside the per-game loop) so the whole
    # family (_pressing / _progression / _buildup) is O(total rows), not O(games x total rows).
    bu_groups = group_rows(bu, "game_id")
    post_groups = group_rows(post, "game_id")
    sgk_groups = group_rows(sgk, "game_id")
    game_groups = group_rows(a, "game_id")

    rows: list[dict] = []
    for game_id in pd.unique(a["game_id"]):
        ga = game_groups.get(game_id)
        game_bu = bu_groups.get(game_id)
        game_post = post_groups.get(game_id)
        game_sgk = sgk_groups.get(game_id)
        for team_id in pd.unique(ga["team_id"].dropna()):
            rows.append(_buildup_row(game_id, team_id, game_bu, game_post, game_sgk))
    return pd.DataFrame(rows, columns=_BUILDUP_COLUMNS)


def _possession_frame(a: pd.DataFrame, spells: pd.DataFrame, params) -> pd.DataFrame:
    """spells + per-possession features (max_x, had_shot, first/second type, has_switch) + next fields."""
    ranked = a.assign(_rank=a.groupby(["game_id", "possession_id"], sort=False).cumcount())
    second_type = ranked[ranked["_rank"] == 1].set_index(["game_id", "possession_id"])["type_id"].rename("second_type")
    switch = a[(a["type_id"] == _PASS) & ((a["end_y"] - a["start_y"]).abs() > params.switch_min_lateral_m)]
    has_switch_ids = set(map(tuple, switch[["game_id", "possession_id"]].drop_duplicates().to_numpy()))

    feat = (
        a.assign(_shot=a["type_id"].isin(_SHOT_TYPE_IDS))
        .groupby(["game_id", "possession_id"], sort=False)
        .agg(max_x=("start_x", "max"), had_shot=("_shot", "max"), first_type=("type_id", "first"))
        .reset_index()
    )
    sf = spells.merge(feat, on=["game_id", "possession_id"], how="left").merge(
        second_type, on=["game_id", "possession_id"], how="left"
    )
    sf["is_short_goalkick"] = (sf["first_type"] == _GOALKICK) & (sf["second_type"] == _PASS)
    sf["has_switch"] = [(g, p) in has_switch_ids for g, p in zip(sf["game_id"], sf["possession_id"], strict=True)]

    sf = sf.sort_values(["game_id", "period_id", "start_time", "possession_id"])
    grp = sf.groupby(["game_id", "period_id"], sort=False)
    sf["next_team"] = grp["team_id"].shift(-1)
    sf["next_had_shot"] = grp["had_shot"].shift(-1)
    sf["next_first_type"] = grp["first_type"].shift(-1)
    return sf


def _classify_buildups(sf: pd.DataFrame, build_up_zone_max_x, half_x, fq_x) -> pd.DataFrame:
    bu = sf[sf["is_open_play"] & (sf["start_x_ltr"] < build_up_zone_max_x)].copy()
    if bu.empty:
        bu["_state"] = pd.Series([], dtype=object)
        return bu
    lost_to_opp = ids_differ(bu["team_id"], bu["next_team"]).fillna(False).to_numpy()
    max_x = bu["max_x"].to_numpy(dtype="float64")
    last_x = bu["last_x_ltr"].to_numpy(dtype="float64")
    # ``.eq`` yields a clean bool Series (a missing "next possession" compares False), avoiding the
    # object-dtype ``fillna`` downcast warning.
    next_shot = bu["next_had_shot"].eq(True).to_numpy()
    next_is_int = bu["next_first_type"].eq(_INTERCEPTION).to_numpy()
    own_half_loss = lost_to_opp & (last_x < half_x)
    bu["_state"] = np.select(
        [
            max_x >= fq_x,
            max_x >= half_x,
            lost_to_opp & next_shot,
            own_half_loss & next_is_int,
            own_half_loss,
        ],
        ["final_quarter", "next_phase", "led_opp_shot", "opp_int_own_half", "opp_won_own_half"],
        default="stayed_phase_one",
    )
    return bu


def _post_regain_summary(a: pd.DataFrame, sf: pd.DataFrame) -> pd.DataFrame:
    """Per-recovery-possession security summary, tagged with the owning team."""
    recov = sf[sf["is_recovery"]][["game_id", "possession_id", "team_id"]]
    if recov.empty:
        return pd.DataFrame(
            columns=["game_id", "team_id", "forward_first", "first_pass_failed", "second_pass_completed"]
        )
    rp = a.merge(recov[["game_id", "possession_id"]], on=["game_id", "possession_id"], how="inner")
    rp = rp.sort_values(["game_id", "possession_id", "time_seconds", "action_id"])
    g = rp.groupby(["game_id", "possession_id"], sort=False)
    forward = (g["end_x"].first() > g["start_x"].first()).rename("forward_first")

    passes = rp[rp["type_id"] == _PASS].copy()
    passes["_prank"] = passes.groupby(["game_id", "possession_id"], sort=False).cumcount()
    first_pass = passes[passes["_prank"] == 0].set_index(["game_id", "possession_id"])["result_id"]
    second_pass = passes[passes["_prank"] == 1].set_index(["game_id", "possession_id"])["result_id"]

    s = recov.set_index(["game_id", "possession_id"]).copy()
    s["forward_first"] = forward.reindex(s.index)
    s["first_pass_failed"] = first_pass.reindex(s.index).eq(_FAIL)
    s["second_pass_completed"] = second_pass.reindex(s.index).eq(_SUCCESS).where(second_pass.reindex(s.index).notna())
    return s.reset_index()


def _buildup_row(game_id, team_id, game_bu, game_post, game_sgk) -> dict:
    row: dict = {"game_id": game_id, "team_id": team_id}

    # --- build-up outcome taxonomy ---
    team_bu = game_bu[ids_match(game_bu["team_id"], team_id).to_numpy()] if len(game_bu) else game_bu
    counts = team_bu["_state"].value_counts() if len(team_bu) else pd.Series(dtype=int)
    for state, col in _BUILDUP_STATE_COLUMNS.items():
        row[col] = int(counts.get(state, 0))
    n_bu = len(team_bu)
    if n_bu > 0:
        success = row["buildup_final_quarter"] + row["buildup_next_phase"]
        row["buildup_success_pct"] = float(success / n_bu)
    else:
        row["buildup_success_pct"] = np.nan

    # --- post-regain security ---
    tp = game_post[ids_match(game_post["team_id"], team_id).to_numpy()] if len(game_post) else game_post
    n_recov = len(tp)
    if n_recov > 0:
        row["post_regain_forward_first_pct"] = float(tp["forward_first"].mean())
        row["post_regain_failed_first_passes"] = int(tp["first_pass_failed"].sum())
        sp = tp["second_pass_completed"].dropna()
        row["post_regain_second_pass_pct"] = float(sp.mean()) if len(sp) else np.nan
    else:
        row["post_regain_forward_first_pct"] = np.nan
        row["post_regain_failed_first_passes"] = 0
        row["post_regain_second_pass_pct"] = np.nan

    # --- switch-conditioned press (over the OPPONENT's short goal kicks) ---
    opp_sgk = (
        game_sgk[ids_differ(game_sgk["team_id"], pd.Series(team_id, index=game_sgk.index)).to_numpy()]
        if len(game_sgk)
        else game_sgk
    )
    n_sgk = len(opp_sgk)
    row["switch_press_n"] = n_sgk
    if n_sgk > 0:
        regained = ids_match(opp_sgk["next_team"], team_id).to_numpy()
        no_switch = ~opp_sgk["has_switch"].to_numpy(dtype=bool)
        row["switch_press_success_pct"] = float(np.mean(regained & no_switch))
    else:
        row["switch_press_success_pct"] = np.nan
    return row
