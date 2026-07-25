"""Owner-gated e2e: TF-51 defensive-credit family on a real GS match with a fitted xT + injected xG.

SCOPE (P-8): this is a PLUMBING / SANITY smoke -- it asserts the family runs end-to-end on real data
and produces sane sign/magnitude distributions. It is NOT xG or xT accuracy validation: the xG is a
crude distance heuristic and xT is fit on the single match. Accuracy validation is the owner-run
SkillCorner cross-check (spec section 12), not this test.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_MATCH = "10502"
_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _TOKEN, reason="owner-tier Gradient Sports data (PINING_FOR_THE_DATA_TOKEN)"),
]


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_loader_pining", str(Path(__file__).parents[2] / "scripts" / "_loader_pining.py")
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _fit_xt(actions):
    from silly_kicks.xthreat import ExpectedThreat

    return ExpectedThreat().fit(actions)


def test_defensive_credit_family_on_real_gs_match():
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking import (
        DEFENSIVE_CREDIT_RULES,
        add_defensive_credit,
        compute_bravery,
        compute_defensive_credits,
    )

    L = _load_loader()
    _prov, _m, actions, frames, _home = next(
        iter(L.load_matches(providers=["gradientsports"], match_ids={"gradientsports": [_MATCH]}))
    )
    # injected xG proxy (no lakehouse xG here): a crude shot-distance heuristic -- SANITY only, P-8.
    shot = actions["type_id"] == spadlconfig.actiontype_id["shot"]
    dist = np.hypot(105.0 - actions["start_x"], 34.0 - actions["start_y"])
    actions = actions.copy()
    actions["xg"] = np.where(shot, np.clip(0.4 * np.exp(-dist / 12.0), 0.0, 1.0), np.nan)
    xt = _fit_xt(actions)

    long = compute_defensive_credits(actions, frames, xg_column="xg", xt=xt)
    assert not long.empty, "expected some defensive credit rows on a real match"
    assert set(long["rule"]).issubset(set(DEFENSIVE_CREDIT_RULES))
    assert long["signed_value"].abs().max() <= 1.0  # sane magnitudes (xG in [0,1], xT small)

    agg = add_defensive_credit(actions, frames, xg_column="xg", xt=xt)  # P-2: no home_team_id; on-target via TF-48
    assert (agg["n_defensive_credits"].fillna(0) >= 0).all()
    assert np.isfinite(agg["defensive_credit_net"].to_numpy()).all()  # always finite

    brav = compute_bravery(actions)
    assert (brav["bravery_pct_known_domain"].dropna().between(0.0, 1.0)).all()
    assert brav["n_set_piece_crosses_faced"].sum() >= 0  # set-piece gap exposed


def test_defensive_credit_v2_acceptance_numbers(capsys):
    """TF-51 v2 acceptance table (spec section 9): the e2e's REAL deliverable is quantitative -- each
    number decides whether an item shipped. Owner runs with `-s` to see the printed table; the
    assert-bounds encode the section-9 fail conditions (GK-flag coverage 0 -> the GK exclusion is
    vacuous; fallback dominance / lane==0 -> Item 2 near-vacuous; no between_lines -> Item 3 gate
    never fires; press source all 'computed' or single-sign -> the distance gate / axis is wrong)."""
    import time

    import numpy as np
    import pandas as pd

    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking import add_press_commitment, compute_defensive_credits, link_actions_to_frames
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl
    from silly_kicks.tracking.defensive_credit import DefensiveCreditParams
    from silly_kicks.tracking.defensive_credit._chaining import with_possessions
    from silly_kicks.tracking.defensive_credit._line_break_signal import precompute_line_break_between_lines
    from silly_kicks.tracking.defensive_credit._resolution import resolve_responsible_defenders

    L = _load_loader()
    _prov, _m, actions, frames, _home = next(
        iter(L.load_matches(providers=["gradientsports"], match_ids={"gradientsports": [_MATCH]}))
    )
    _shot = actions["type_id"] == spadlconfig.actiontype_id["shot"]
    _dist = np.hypot(105.0 - actions["start_x"], 34.0 - actions["start_y"])
    actions = actions.copy()
    actions["xg"] = np.where(_shot, np.clip(0.4 * np.exp(-_dist / 12.0), 0.0, 1.0), np.nan)
    xt = _fit_xt(actions)

    # Per-match wall-time (Task 2b Step 3): the Item-3 line-break precompute clusters candidate passes
    # with Ward; a clustering regression (e.g. per-action re-linking / clustering every action) inflates
    # this. Reported, not asserted -- CI's deterministic call-count guard is the hard gate.
    _t0 = time.perf_counter()
    long = compute_defensive_credits(actions, frames, xg_column="xg", xt=xt)
    _wall = time.perf_counter() - _t0

    # --- Item 2: lane-geometry shot_block ---
    sb = long[long["rule"] == "shot_block"]
    resolution_counts = sb["resolution"].value_counts().to_dict()
    gk_players = set(pd.Series(frames.loc[frames["is_goalkeeper"].astype(bool), "player_id"]).dropna().unique())
    n_gk_credited = int(sb["player_id"].isin(gk_players).sum())
    gk_by_team = frames.loc[frames["is_goalkeeper"].astype(bool)].groupby("team_id")["player_id"].nunique().to_dict()

    # v1-equivalent (nearest-to-origin) vs v2 (lane) per blocked shot -> % of credited players CHANGED
    act = with_possessions(actions).reset_index(drop=True)
    pointers = link_actions_to_frames(act, frames)[0]
    fid_by_pos = (
        pointers.drop_duplicates("action_id")
        .set_index("action_id")["frame_id"]
        .reindex(act["action_id"].to_numpy())
        .to_numpy()
    )
    flip_series = acting_team_attacks_rtl(act, frames)
    _shot_id = spadlconfig.actiontype_id["shot"]
    blocked_mask = act["shot_blocked"].fillna(False).astype(bool) & (act["type_id"] == _shot_id)
    n_changed = n_compared = n_v1_rows = 0
    p = DefensiveCreditParams()
    for pos in np.where(blocked_mask.to_numpy())[0]:
        a = act.iloc[int(pos)]
        fid = fid_by_pos[pos]
        fid = None if pd.isna(fid) else int(fid)
        flip = bool(flip_series.iloc[int(pos)])
        kw = dict(
            anchor_x=a["start_x"],
            anchor_y=a["start_y"],
            acting_team_id=a["team_id"],
            params=p,
            frame_id=fid,
            flip=flip,
        )
        v1 = resolve_responsible_defenders(act, frames, mode="nearest", **kw)
        v2 = resolve_responsible_defenders(act, frames, mode="lane_blocker", **kw)
        if not v1.empty:
            n_v1_rows += 1
        if v1.empty and v2.empty:
            continue
        n_compared += 1
        v1p = None if v1.empty else v1["player_id"].iloc[0]
        v2p = None if v2.empty else v2["player_id"].iloc[0]
        if v1p != v2p:
            n_changed += 1
    pct_changed = (n_changed / n_compared) if n_compared else 0.0

    # --- Item 3: line-break state distribution + firing ---
    lb = pd.Series(
        precompute_line_break_between_lines(act, frames, fid_by_pos=fid_by_pos, flip_by_pos=flip_series.to_numpy())
    )
    n_true = int((lb == True).sum())  # noqa: E712
    n_false = int((lb == False).sum())  # noqa: E712  (computed-False incl. short-circuit-0)
    n_na = int(lb.isna().sum())
    n_through_fires = int((long["rule"] == "failed_marking_through_ball").sum())

    # --- Item 5: press-commitment cue ---
    pc = add_press_commitment(actions, frames)
    src_counts = pc["press_commitment_source"].value_counts().to_dict()
    vals = pc["press_commitment"].dropna()
    n_pos = int((vals > 0).sum())
    n_neg = int((vals < 0).sum())

    with capsys.disabled():
        print("\n=== TF-51 v2 acceptance table (GS match", _MATCH, ") ===")
        print(
            f"compute_defensive_credits wall-time: {_wall:.2f}s ({len(actions)} actions) "
            "-- a Ward-clustering regression inflates this"
        )
        print("Item 2 shot_block  resolution:", resolution_counts, "| total:", len(sb))
        print(
            "Item 2 credited-player CHANGED vs v1(nearest): "
            f"{n_changed}/{n_compared} = {pct_changed:.1%} (v1 rows: {n_v1_rows})"
        )
        print("Item 2 blocks credited to a GK:", n_gk_credited, "| per-team GK coverage:", gk_by_team)
        print("Item 3 line-break True/False/<NA>:", n_true, n_false, n_na, "| through fires:", n_through_fires)
        print("Item 5 press_commitment_source:", src_counts, "| sign +/-:", n_pos, "/", n_neg)

    # section-9 fail conditions
    assert sum(gk_by_team.values()) > 0, "N5: is_goalkeeper flag all-False -> GK-credit=0 would be vacuous"
    assert n_gk_credited == 0, "Item 2: a shot_block was credited to a GK (a save, not a block)"
    assert resolution_counts.get("lane", 0) > 0, "Item 2 near-vacuous: nothing resolved on the lane"
    assert n_true > 0, "Item 3: no between_lines line-break detected on a full match -> gate never fires"
    assert set(src_counts) != {"computed"}, "Item 5: press source all 'computed' -> the distance gate is missing"
    assert n_pos > 0 and n_neg > 0, "Item 5: press_commitment is single-sign -> axis or slope is wrong"
