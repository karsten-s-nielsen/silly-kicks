#!/usr/bin/env python
"""Validate the shipped SkillCorner keeper-origin resolver on real pining data (ADR-024).

S1-S4 shipped in 4.37.0/PR-S104; this driver VALIDATES them on the real pining SkillCorner corpus
and characterizes the off-pitch / behind-the-goal-line rows that feed the gr_x decision (Phase B of
the keeper-box geometry & detection-quality cycle). It re-uses the library resolver
(``resolve_gk_geometry``) and goal map (``resolve_defended_goals``); it never re-implements
resolution.

Domain: goal-kicks (the origin-imputation target of ``resolve_gk_geometry``) plus open-play passes
taken by a goalkeeper -- the GK-distribution domain. ``gr_x`` is distance from the acting team's
DEFENDED goal, resolved per row via the ADR-055 goal map (never ``home_team_id``).

Sources every provider through pining-for-the-data; the downloaded folders are not an input.
Adopts ``scripts/_driver.py`` (``for_each``) and stamps provenance via ``scripts/_provenance.py``.

Usage:
    python scripts/validate_skillcorner_keeper_origin.py --out DIR [--match-ids-json IDS.json]
        [--max-per-provider N] [--allow-dirty]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from _driver import for_each, reconcile
from _offpitch import off_pitch_mask
from _provenance import git_provenance, require_clean_tree

_GOALKICK_TYPE_ID = 22  # SPADL goalkick type_id
_SHARD_SCHEMA_VERSION = "skc-keeper-origin-2"  # bumped: added is_goalkick + raw_native_out_of_region (ADR-052)

EXPECTED_COLS = [
    "provider",
    "match_id",
    "action_id",
    "is_goalkick",
    "xt_gk_origin_source",
    "origin_x",
    "origin_y",
    "in_own_box",
    "gr_x",
    "y",
    "in_y_band",
    "is_behind_line",
    "is_gross_offpitch",
    "is_visible",
    "raw_native_out_of_region",
]


def _gk_players_by_team(frames: pd.DataFrame) -> dict:
    """Map team_id -> set of goalkeeper player_ids observed in the frames."""
    if "is_goalkeeper" not in frames.columns:
        return {}
    gks = frames[frames["is_goalkeeper"].fillna(False).astype(bool)]
    out: dict = {}
    for team_id, grp in gks.groupby("team_id"):
        out[team_id] = set(grp["player_id"].dropna().tolist())
    return out


def _gk_distribution_mask(actions: pd.DataFrame, gk_by_team: dict) -> np.ndarray:
    """GK-distribution domain: goal-kicks, plus passes whose actor is a known goalkeeper."""
    from silly_kicks.id_compat import ids_match

    is_goalkick = ids_match(actions["type_id"], _GOALKICK_TYPE_ID).to_numpy(dtype=bool)
    is_gk_actor = np.zeros(len(actions), dtype=bool)
    if gk_by_team:
        players = actions["player_id"].tolist()
        teams = actions["team_id"].tolist()
        for i, (team, player) in enumerate(zip(teams, players, strict=False)):
            is_gk_actor[i] = player in gk_by_team.get(team, set())
    return is_goalkick | is_gk_actor


def _visible_at_action(frames: pd.DataFrame, links: pd.DataFrame, gk_by_team: dict, action_id, team_id) -> object:
    """Whether the acting team's goalkeeper is DETECTED at the action's linked frame.

    Returns True / False / pd.NA (NA when there is no linked frame or no GK row to read).
    """
    if "visibility" not in frames.columns:
        return pd.NA
    row = links[links["action_id"] == action_id]
    if row.empty or pd.isna(row["frame_id"].iloc[0]):
        return pd.NA
    frame_id = row["frame_id"].iloc[0]
    fr = frames[frames["frame_id"] == frame_id]
    gk_players = gk_by_team.get(team_id, set())
    gk_rows = fr[fr["player_id"].isin(gk_players)]
    if gk_rows.empty:
        return pd.NA
    return bool(gk_rows["visibility"].fillna(False).astype(bool).any())


def measure_match(
    provider: str, match_id: str, actions: pd.DataFrame, frames: pd.DataFrame, home_team_id
) -> pd.DataFrame:
    """One row per GK-distribution action.

    Emitted origin fields are the SHIPPED resolution: distrust is decided per provider via
    ``native_origin_is_trusted`` (SkillCorner goal-kicks are distrusted, so the raw broadcast-ball
    origin is imputed to ``goalkick_prior``/``tracking_gk``). ``raw_native_out_of_region`` is the
    DIAGNOSTIC before-picture (distrust OFF) -- the artifact the resolver corrects, reported not gated.
    """
    from silly_kicks.id_compat import ids_match
    from silly_kicks.tracking import link_actions_to_frames
    from silly_kicks.tracking._geometry import GOAL_Y, in_penalty_area_goal_relative
    from silly_kicks.tracking._gk_geometry import native_origin_is_trusted, resolve_gk_geometry

    gk_by_team = _gk_players_by_team(frames)
    dom_mask = _gk_distribution_mask(actions, gk_by_team)
    dom = actions[dom_mask]
    if dom.empty:
        return pd.DataFrame(columns=EXPECTED_COLS)

    distrust = not native_origin_is_trusted(provider)
    geom = resolve_gk_geometry(actions, frames=frames, distrust_native_origin=distrust)  # SHIPPED
    geom_raw = resolve_gk_geometry(actions, frames=frames, distrust_native_origin=False)  # raw before
    links, _report = link_actions_to_frames(actions, frames)

    geom_dom = geom.reindex(dom.index)
    raw_dom = geom_raw.reindex(dom.index)
    ox = geom_dom["origin_x"].to_numpy(dtype=float)
    oy = geom_dom["origin_y"].to_numpy(dtype=float)
    osrc = geom_dom["origin_source"].to_numpy()
    rox = raw_dom["origin_x"].to_numpy(dtype=float)
    roy = raw_dom["origin_y"].to_numpy(dtype=float)
    gk_action = ids_match(dom["type_id"], _GOALKICK_TYPE_ID).to_numpy(dtype=bool)

    rows = []
    for i, (_idx, a) in enumerate(dom.iterrows()):
        origin_x = float(ox[i]) if not np.isnan(ox[i]) else np.nan
        origin_y = float(oy[i]) if not np.isnan(oy[i]) else np.nan
        # `resolve_gk_geometry` emits origins in ACTION-LTR: the acting team attacks x=105 and DEFENDS
        # the goal at x=0, so the goal-relative x IS origin_x. Do NOT route through the frame goal map
        # (`resolve_defended_goals` is home-attacks-right): for an away-team action the two frames are
        # 105 m apart, gr_x would flip to ~100, and every away goal-kick would read out-of-box. That is
        # the ADR-028 defect this probe exists to measure -- caught here on real SkillCorner data, where
        # the frame-goal_map version scored 28.6% own-box against the correct 100%.
        gr_x = origin_x
        in_band = bool(abs(origin_y - GOAL_Y) <= _half_width()) if not np.isnan(origin_y) else False
        in_box = bool(in_penalty_area_goal_relative(gr_x, origin_y)) if not np.isnan(gr_x) else False
        raw_oor: object = pd.NA  # goal-kicks only: is the RAW native origin out of the own box?
        if bool(gk_action[i]) and not np.isnan(rox[i]):
            raw_oor = not bool(in_penalty_area_goal_relative(rox[i], roy[i]))
        rows.append(
            {
                "provider": provider,
                "match_id": str(match_id),
                "action_id": a["action_id"],
                "is_goalkick": bool(gk_action[i]),
                "xt_gk_origin_source": osrc[i],
                "origin_x": origin_x,
                "origin_y": origin_y,
                "in_own_box": in_box,
                "gr_x": gr_x,
                "y": origin_y,
                "in_y_band": in_band,
                "is_behind_line": bool(gr_x < 0.0) if not np.isnan(gr_x) else False,
                "is_gross_offpitch": bool(off_pitch_mask(np.array([origin_x]), np.array([origin_y]))[0]),
                "is_visible": _visible_at_action(frames, links, gk_by_team, a["action_id"], a["team_id"]),
                "raw_native_out_of_region": raw_oor,
            }
        )
    return pd.DataFrame(rows, columns=EXPECTED_COLS)


def _half_width() -> float:
    import silly_kicks.spadl.config as spadlconfig

    return float(spadlconfig.penalty_area_half_width)


def offpitch_rate(frame: pd.DataFrame) -> float:
    """S1 gross-off-pitch rate over the GK-distribution rows (ADR-024 S1)."""
    if frame.empty:
        return 0.0
    return float(frame["is_gross_offpitch"].fillna(False).astype(bool).mean())


def out_of_region_goalkick_rate(frame: pd.DataFrame) -> float:
    """GATED S4: fraction of goal-kicks whose SHIPPED-RESOLVED origin is outside the own box.

    ~0 for a provider the resolver handles (SkillCorner goal-kicks are distrusted -> imputed in-box,
    i.e. the ADR-024 "~=100% own-box" acceptance); a provider it does NOT handle keeps its native
    origins and trips this. The raw before-picture is the separate diagnostic below, never this."""
    gk = frame[frame["is_goalkick"].fillna(False).astype(bool)]
    if gk.empty:
        return 0.0
    return float((~gk["in_own_box"].fillna(False).astype(bool)).mean())


def raw_native_goalkick_out_of_region_rate(frame: pd.DataFrame) -> float:
    """DIAGNOSTIC (never gated): fraction of goal-kicks whose RAW native origin is out of region --
    the broadcast-ball artifact the resolver corrects (SkillCorner ~0.6). A handled provider
    legitimately has a high raw rate, which is exactly why this is reported, not gated."""
    raw = frame["raw_native_out_of_region"].dropna()
    if raw.empty:
        return 0.0
    return float(raw.astype(bool).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate SkillCorner keeper-origin resolution on pining data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-ids-json", default=None)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    from _loader_pining import load_matches

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    match_ids = {"skillcorner": json.load(open(args.match_ids_json))} if args.match_ids_json else None
    items = load_matches(providers=["skillcorner"], match_ids=match_ids, max_per_provider=args.max_per_provider)

    res = for_each(
        items,
        key=lambda m: ("skillcorner", str(m[1])),
        work=lambda m: measure_match(*m),
        shard_root=out / "shards",
        token_inputs={"schema": _SHARD_SCHEMA_VERSION, "driver": "skillcorner-keeper-origin"},
    )
    combined = reconcile(res.shard_dir, out / "skillcorner_keeper_origin.parquet", tag="all")

    manifest = {
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "platform": prov["platform"],
        "machine": prov["machine"],
        "n_rows": len(combined),
        "offpitch_rate": offpitch_rate(combined),
        "out_of_region_goalkick_rate": out_of_region_goalkick_rate(combined),  # GATED (resolved)
        "raw_native_goalkick_out_of_region_rate": raw_native_goalkick_out_of_region_rate(combined),  # DIAGNOSTIC
    }
    (out / "manifest_all.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out / 'skillcorner_keeper_origin.parquet'} ({len(combined)} rows)", flush=True)


if __name__ == "__main__":
    main()
