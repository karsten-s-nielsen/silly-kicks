"""Crosser-anchored opportunity-row builder for the xCross causal harness (ADR-015).

A per-(game,period) spell state-machine: one row per continuous wide-area possession-spell, anchored
at entry (the paper's sender-level unit). The spell end serves as the dedup boundary AND the ceiling
on the treatment window (R3-M1):
  Z = 1 iff a possessing-team cross occurs in (entry, min(entry + EXPOSURE_WINDOW_SECONDS, spell_end)];
      the fixed T cap keeps Z-exposure bounded (no spell-length confounding -- Y's window is already
      fixed, so clamping to spell_end adds no duration->Y path), and the spell_end cap prevents
      misattributing a cross from a LATER re-possession phase to this opportunity.
  Y = a possessing-team shot in (anchor, anchor + OUTCOME_WINDOW_SECONDS], anchor = t_cross for
      treated (strictly post-cross -> no reverse-direction leakage, R2-M1) else entry for controls;
      Y is NOT possession-clamped (documented modeling choice -- treated/control windows time-shifted).
X = the 7 paper confounders (imported from _xcross_attempt._CONFOUNDERS -- single source, R2-M2) + 6
GK columns; ball-geometry features are excluded (surface-model inputs, not paper confounders). Pure;
no I/O. Reuses the shipped xCross domain/carrier/feature helpers so the matched corpus is the model's
training domain by construction. Dedup R2-M1: a new spell starts only on a possession break or a
wide-area domain exit; a mid-spell carrier hand-off stays one row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl import config as _spc
from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
from silly_kicks.tracking._id_compat import ids_match, same_id
from silly_kicks.tracking._xcross_attempt import (
    _ADVANCE_M,
    _CONFOUNDERS,
    XCROSS_FEATURE_NAMES_FAITHFUL,
    _build_goal_map,
    _has_results,
    _in_wide_area,
    extract_xcross_features,
)

# X split (M3 + R2-M2): the 7 paper confounders are the SINGLE-SOURCE _CONFOUNDERS (not re-literal'd);
# ball_r/theta/speed are surface-model inputs, NOT paper confounders, and are excluded from the causal X.
PAPER_CONFOUNDERS = list(_CONFOUNDERS)
GK_BLOCK = [c for c in XCROSS_FEATURE_NAMES_FAITHFUL if c.startswith("gk_")]

# Pre-registered windows (named + asserted). The treatment/outcome windows are bounded by a fixed cap
# (R2-H3), so they are NOT a function of spell length; MAX_SPELL_SECONDS only bounds the dedup machine.
MAX_SPELL_SECONDS = 30.0  # dedup cap: split a pathological never-closing in-domain run
EXPOSURE_WINDOW_SECONDS = 8.0  # T: Z = cross in (entry, min(entry+T, spell_end)]
OUTCOME_WINDOW_SECONDS = 6.0  # W: Y = shot in (anchor, anchor+W]

_PROV_COLS = [
    "game_id",
    "period_id",
    "entry_frame_id",
    "entry_time",
    "end_time",
    "spell_duration_seconds",
    "possessing_team",
    "carrier_resolved",
]


def build_opportunities(frames, actions, *, home_team_id, model_metadata, advance_m=_ADVANCE_M) -> pd.DataFrame:
    """Return one row per continuous wide-area possession-spell: the 7 paper confounders + 6 GK
    columns, treatment ``Z``, outcome ``Y``, and provenance. Pure; no I/O.

    Examples
    --------
    >>> import pandas as pd
    >>> from tests.causal._fixtures import META, WIDE, actions, frames  # doctest: +SKIP
    >>> f = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE})  # doctest: +SKIP
    >>> build_opportunities(f, actions([]), home_team_id=5, model_metadata=META).shape[0]  # doctest: +SKIP
    1
    """
    cross_types = tuple(model_metadata.get("cross_types", ("cross",)))
    carrier_params = dict(model_metadata.get("carrier_params", {}))
    carrier = infer_ball_carrier(frames, **carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)
    score_fn = None
    if _has_results(actions) and home_team_id is not None:
        from silly_kicks.tracking._ghost_gk import _build_score_lookup

        score_fn = _build_score_lookup(actions, home_team_id)

    spells: list[dict] = []
    for (gid, per), g in poss.groupby(["game_id", "period_id"], sort=False):
        g = g.sort_values(["time_seconds", "frame_id"])
        frame_keys = list(dict.fromkeys(zip(g["frame_id"].tolist(), g["time_seconds"].tolist(), strict=True)))
        spell: dict | None = None
        for fid, t in frame_keys:
            grp = g[g["frame_id"] == fid]
            team, goal_x, in_dom = _frame_domain_state(grp, goal_map, gid, per, advance_m)
            if (  # spell continues iff same team, still in domain, under the dedup cap (R2-L2: same_id)
                spell is not None
                and in_dom
                and same_id(team, spell["team"])
                and (t - spell["entry_time"]) <= MAX_SPELL_SECONDS
            ):
                spell["end_time"], spell["end_frame_id"] = float(t), fid
                continue
            if spell is not None:
                spells.append(spell)
                spell = None
            if in_dom:
                spell = dict(
                    gid=gid,
                    per=per,
                    team=team,
                    goal_x=goal_x,
                    grp=grp,
                    entry_frame_id=fid,
                    entry_time=float(t),
                    end_time=float(t),
                    end_frame_id=fid,
                )
        if spell is not None:
            spells.append(spell)

    rows = [_row(sp, actions, cross_types, score_fn, home_team_id) for sp in spells]
    cols = PAPER_CONFOUNDERS + GK_BLOCK + _PROV_COLS + ["Z", "Y"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _frame_domain_state(grp, goal_map, gid, per, advance_m):
    in_poss = grp["team_in_possession"].dropna()
    if in_poss.empty:
        return None, None, False
    poss_team = in_poss.iloc[0]
    ball = grp[grp["is_ball"]]
    if "ball_state" in grp.columns and len(ball) and str(ball["ball_state"].iloc[0]) == "dead":
        return poss_team, None, False
    non_ball = grp[~grp["is_ball"].astype(bool)]
    defending = [d for d in non_ball["team_id"].dropna().unique() if not same_id(d, poss_team)]
    if not defending:
        return poss_team, None, False
    goal_x = goal_map.get((gid, per, defending[0]))
    if goal_x is None:
        return poss_team, None, False
    bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
    by = float(ball["y"].iloc[0]) if len(ball) else np.nan
    return poss_team, goal_x, _in_wide_area(bx, by, goal_x, advance_m)


def _row(sp, actions, cross_types, score_fn, home_team_id) -> dict:
    grp, gid, per, team, goal_x = sp["grp"], sp["gid"], sp["per"], sp["team"], sp["goal_x"]
    carrier_s = grp["ball_carrier_player_id"].dropna()
    carrier_pid = carrier_s.iloc[0] if not carrier_s.empty else None
    non_ball = grp[~grp["is_ball"].astype(bool)]
    defending = [d for d in non_ball["team_id"].dropna().unique() if not same_id(d, team)]
    sd = np.nan
    if score_fn is not None:
        # R2-L1: _build_score_lookup returns a _zero callback when no goals -> raw is never None/NaN.
        raw = score_fn(gid, sp["entry_time"])  # home - away
        sd = float(raw) if same_id(team, home_team_id) else -float(raw)
    feats = extract_xcross_features(
        grp, gk_team_id=defending[0], goal_x=goal_x, carrier_player_id=carrier_pid, score_differential=sd
    ).iloc[0]
    row = {c: float(feats[c]) for c in PAPER_CONFOUNDERS + GK_BLOCK}
    entry = sp["entry_time"]
    z, t_cross = _label_treatment(actions, gid, per, team, cross_types, entry, sp["end_time"])
    anchor = t_cross if z else entry
    row.update(
        game_id=gid,
        period_id=per,
        entry_frame_id=sp["entry_frame_id"],
        entry_time=entry,
        end_time=sp["end_time"],
        spell_duration_seconds=sp["end_time"] - entry,
        possessing_team=team,
        carrier_resolved=carrier_pid is not None,
        Z=z,
        Y=_label_outcome(actions, gid, per, team, anchor),
    )
    return row


def _team_period_action_times(actions, gid, per, team, type_names) -> np.ndarray:
    type_ids = {_spc.actiontype_id[n] for n in type_names}
    sel = (  # ids_match: dtype-safe action<->frame team/game id seam (ADR-019)
        ids_match(actions["game_id"], gid)
        & (actions["period_id"] == per)
        & ids_match(actions["team_id"], team)
        & actions["type_id"].isin(type_ids)
    )
    return np.sort(actions.loc[sel, "time_seconds"].to_numpy(dtype=float))


def _label_treatment(actions, gid, per, team, cross_types, entry, end_time) -> tuple[int, float | None]:
    hi = min(entry + EXPOSURE_WINDOW_SECONDS, end_time)  # R3-M1: clamp the Z-window to possession continuity
    ts = _team_period_action_times(actions, gid, per, team, cross_types)
    win = ts[(ts > entry) & (ts <= hi)]
    return (1, float(win[0])) if len(win) else (0, None)


def _label_outcome(actions, gid, per, team, anchor) -> int:
    ts = _team_period_action_times(actions, gid, per, team, ("shot", "shot_freekick", "shot_penalty"))
    return int(bool(((ts > anchor) & (ts <= anchor + OUTCOME_WINDOW_SECONDS)).any()))
