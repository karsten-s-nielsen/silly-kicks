"""Pressing / defensive KPI family (TF-52 Section 4.1).

Per ``(game_id, team_id)``: PPDA, defensive intensity, time-to-defensive-action, time-to-recovery,
recoveries (+within-Ns %), and the configurable counter-press window (seconds XOR passes).

Opponent-relative KPIs (PPDA) combine our rows with the opponent's rows reflected into our
action-LTR frame (``_orientation``); every other KPI derives from the possession spells (``_possession``)
and per-team action counts. Honest-NaN on empty denominators (never a fabricated 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import ids_match
from silly_kicks.spadl import config as spadlconfig

from ._orientation import reflect_into_team_frame

_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]

_PRESSING_COLUMNS = [
    "game_id",
    "team_id",
    "ppda",
    "defensive_intensity",
    "time_to_defensive_action_s",
    "time_to_recovery_s",
    "recoveries",
    "recoveries_within_ns_pct",
    "counterpress_regains",
    "counterpress_regain_pct",
]


def compute_pressing_kpis(
    actions_ctx: pd.DataFrame, spells: pd.DataFrame, minutes: pd.DataFrame, *, params
) -> pd.DataFrame:
    """One row per ``(game_id, team_id)`` with the pressing KPIs (see module docstring)."""
    def_ids = [spadlconfig.actiontype_id[n] for n in params.defensive_action_types]
    zone_min_x = (1.0 - params.ppda_zone_fraction) * spadlconfig.field_length
    spells_aug = _augment_regain_gap(spells)

    game_groups = group_rows(actions_ctx, "game_id")
    spell_groups = group_rows(spells_aug, ("game_id", "team_id"))
    minute_groups = group_rows(minutes, ("game_id", "team_id"))

    rows: list[dict] = []
    for game_id in pd.unique(actions_ctx["game_id"]):
        ga = game_groups.get(game_id)
        notna = ga["team_id"].notna().to_numpy()
        for team_id in pd.unique(ga["team_id"].dropna()):
            is_team = ids_match(ga["team_id"], team_id).to_numpy()
            team_actions = ga.iloc[is_team]
            opp_actions = ga.iloc[(~is_team) & notna]
            team_spells = spell_groups.get(game_id, team_id)
            mrow = minute_groups.get(game_id, team_id)
            out_min = float(mrow["out_of_possession_min"].iloc[0]) if len(mrow) else np.nan
            rows.append(
                _pressing_row(
                    game_id,
                    team_id,
                    team_actions,
                    opp_actions,
                    team_spells,
                    out_min,
                    def_ids,
                    zone_min_x,
                    params,
                )
            )
    return pd.DataFrame(rows, columns=_PRESSING_COLUMNS)


def _augment_regain_gap(spells: pd.DataFrame) -> pd.DataFrame:
    """Add ``prev_own_end`` (the team's previous possession's end) + ``regain_gap_s`` (out-of-poss time)."""
    s = spells.sort_values(["game_id", "period_id", "start_time", "possession_id"]).copy()
    s["prev_own_end"] = s.groupby(["game_id", "period_id", "team_id"], sort=False)["end_time"].shift(1)
    s["regain_gap_s"] = s["start_time"] - s["prev_own_end"]
    return s


def _pressing_row(
    game_id, team_id, team_actions, opp_actions, team_spells, out_min, def_ids, zone_min_x, params
) -> dict:
    row: dict = {"game_id": game_id, "team_id": team_id}

    # --- PPDA: opponent passes in zone / our defensive actions in zone (common frame) ---
    opp_reflected = reflect_into_team_frame(opp_actions)
    opp_pass_zone = int(((opp_reflected["type_id"] == _PASS) & (opp_reflected["start_x"] >= zone_min_x)).sum())
    our_def_zone = int((team_actions["type_id"].isin(def_ids) & (team_actions["start_x"] >= zone_min_x)).sum())
    row["ppda"] = (opp_pass_zone / our_def_zone) if our_def_zone > 0 else np.nan

    # --- defensive intensity: defensive actions per minute out of possession ---
    n_def = int(team_actions["type_id"].isin(def_ids).sum())
    row["defensive_intensity"] = (n_def / out_min) if (out_min is not None and out_min > 0) else np.nan

    # --- recoveries / within-Ns / time-to-recovery ---
    # ``recoveries`` counts every regain; the TIMING metrics measure only recoveries with a
    # finite regain gap -- a team's first possession can be a recovery (won from the kickoff team)
    # with no prior own loss, so its gap is NA and must not read as "not within".
    recov = team_spells[team_spells["is_recovery"]]
    row["recoveries"] = len(recov)
    gaps = recov["regain_gap_s"].to_numpy(dtype="float64")
    finite = gaps[np.isfinite(gaps)]
    if finite.size > 0:
        row["recoveries_within_ns_pct"] = float(np.mean(finite <= params.counterpress_seconds))
        row["time_to_recovery_s"] = float(np.mean(finite))
    else:
        row["recoveries_within_ns_pct"] = np.nan
        row["time_to_recovery_s"] = np.nan

    # --- counter-press window (v2, seconds XOR passes) ---
    regains, pct = _counterpress(recov, opp_actions, params.counterpress_window)
    row["counterpress_regains"] = regains
    row["counterpress_regain_pct"] = pct

    # --- time to first defensive action after a loss ---
    row["time_to_defensive_action_s"] = _time_to_def_action(team_actions, team_spells, def_ids)
    return row


def _counterpress(recov_spells: pd.DataFrame, opp_actions: pd.DataFrame, window) -> tuple[int, float]:
    """Counter-press regains within ``window`` (seconds arm or opponent-passes-to-regain arm).

    Only recoveries with a finite regain gap (a real prior own loss) qualify -- a first-possession
    recovery has no prior loss to counter-press from.
    """
    recov_spells = recov_spells[recov_spells["regain_gap_s"].notna()]
    n = len(recov_spells)
    if n == 0:
        return 0, np.nan
    if window.seconds is not None:
        within = recov_spells["regain_gap_s"].to_numpy(dtype="float64") <= window.seconds
    else:
        # passes arm: count the opponent's completed passes in each recovery's (loss, regain) interval,
        # PER PERIOD (time_seconds is period-relative, so periods must not be pooled).
        opp = opp_actions[(opp_actions["type_id"] == _PASS) & (opp_actions["result_id"] == _SUCCESS)]
        parts: list[np.ndarray] = []
        for period_id, grp in recov_spells.groupby("period_id", sort=False):
            opp_times = np.sort(opp.loc[opp["period_id"] == period_id, "time_seconds"].to_numpy(dtype="float64"))
            t_loss = grp["prev_own_end"].to_numpy(dtype="float64")
            t_regain = grp["start_time"].to_numpy(dtype="float64")
            cnt = np.searchsorted(opp_times, t_regain, side="right") - np.searchsorted(opp_times, t_loss, side="left")
            parts.append(cnt <= window.passes)
        within = np.concatenate(parts) if parts else np.zeros(0, dtype=bool)
    return int(np.count_nonzero(within)), float(np.mean(within))


def _time_to_def_action(team_actions: pd.DataFrame, team_spells: pd.DataFrame, def_ids) -> float:
    """Mean seconds from a loss (a spell's end) to the team's first defensive action after it."""
    def_actions = team_actions[team_actions["type_id"].isin(def_ids)]
    parts: list[np.ndarray] = []
    for period_id, sp in team_spells.groupby("period_id", sort=False):
        d_times = np.sort(
            def_actions.loc[def_actions["period_id"] == period_id, "time_seconds"].to_numpy(dtype="float64")
        )
        losses = sp["end_time"].to_numpy(dtype="float64")
        idx = np.searchsorted(d_times, losses, side="right")  # first def action strictly after the loss
        has = idx < len(d_times)
        parts.append(d_times[idx[has]] - losses[has])
    deltas = np.concatenate(parts) if parts else np.zeros(0)
    return float(np.mean(deltas)) if deltas.size else np.nan
