"""Owner-gated e2e: TF-49 packing on one full real WC2022 Gradient Sports match.

Hard gates = internal consistency against the 2026-07-16 probe anchors (shipping
non_action-skip rule): receiver resolution 0.9976 +- 0.02 on completed pass-like
rows; GS dribble degenerate rate consistent with post-PR-S116 geometry (<10% --
period-last carries only, NOT the pre-fix 100%; this e2e doubles as the
packing-side verification that PR-S116 landed); secured rate strictly in (0, 1);
the GS dribble in-domain share strictly interior (ballCarryOutcome R/L split);
and (plan B11, PROMOTED to a gate 2026-07-17) the per-action packing_made mean in
a sane band [0.5, 3.0], validated across 4 real WC2022 matches. The MSC
practitioner anchors (~2 bypassed per packing action; ~8 goal-threat points/shot;
67.4% of goals with a packing action in the preceding possession) stay REPORTED,
not gated (league-specific).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from silly_kicks.spadl import config as spadlconfig

_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _TOKEN, reason="owner-tier Gradient Sports data (PINING_FOR_THE_DATA_TOKEN)"),
]

_MATCH_ID = "10503"

_PASS_LIKE = (
    "pass",
    "cross",
    "throw_in",
    "freekick_crossed",
    "freekick_short",
    "corner_crossed",
    "corner_short",
    "goalkick",
)


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_loader_pining", str(Path(__file__).parents[2] / "scripts" / "_loader_pining.py")
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(scope="module")
def gs_match():
    from silly_kicks.tracking import add_packing

    L = _load_loader()
    _prov, _mid, actions, frames, _home = next(
        iter(
            L.load_matches(
                providers=["gradientsports"],
                match_ids={"gradientsports": [_MATCH_ID]},
                tracking_limit=None,
            )
        )
    )
    packed = add_packing(actions, frames)
    return actions, packed


def test_receiver_resolution_rate_within_probe_band(gs_match):
    actions, packed = gs_match
    ids = [spadlconfig.actiontype_id[n] for n in _PASS_LIKE]
    mask = np.isin(actions["type_id"].to_numpy(), ids) & (
        actions["result_id"].to_numpy() == spadlconfig.result_id["success"]
    )
    assert mask.sum() > 300, "unexpectedly few completed pass-like rows"
    rate = packed.loc[mask, "packing_receiver_player_id"].notna().mean()
    assert 0.9776 <= rate <= 1.0, f"GS receiver resolution {rate:.4f} outside 0.9976 +- 0.02"


def test_dribble_degenerate_rate_post_pr_s116(gs_match):
    """TYPE-only filter (geometry property, matching the PR-S116 e2e)."""
    actions, _packed = gs_match
    d = actions[actions["type_id"] == spadlconfig.actiontype_id["dribble"]]
    assert len(d) > 0, "match has no dribbles"
    degenerate = ((d["end_x"] == d["start_x"]) & (d["end_y"] == d["start_y"])).mean()
    assert degenerate < 0.10, f"GS degenerate-dribble rate {degenerate:.1%} -- PR-S116 geometry missing?"


def test_dribble_domain_share_strictly_interior(gs_match):
    """PR-S117 converter fix: BC carries map ballCarryOutcome R->success / L->fail
    (previously EVERY GS dribble fell to the fail default -- structurally off-domain
    for packing's completion gate). The retained share must be strictly interior:
    0 => the outcome field vanished/renamed (mass-fail regression); 1 => the fail
    branch died. Probe grounding: R = 44/66 across 4 WC2022 matches (2026-07-17)."""
    actions, _packed = gs_match
    d = actions[actions["type_id"] == spadlconfig.actiontype_id["dribble"]]
    assert len(d) > 0, "match has no dribbles"
    in_domain = (d["result_id"] == spadlconfig.result_id["success"]).mean()
    assert 0.0 < in_domain < 1.0, f"GS dribble in-domain share {in_domain} not strictly interior"


def test_secured_rate_strictly_interior(gs_match):
    _actions, packed = gs_match
    sec = packed["packing_secured"].dropna()
    assert len(sec) > 50, "too few decided secured labels"
    rate = sec.astype(bool).mean()
    assert 0.0 < rate < 1.0, f"secured rate {rate} not strictly in (0, 1)"


def test_packing_made_mean_in_sane_band(gs_match):
    """Plan B11 / review-minor-10 gate, PROMOTED from reported to gated (2026-07-17).
    Per-action ``packing_made`` mean (over non-NaN in-domain rows) must sit in a sane
    band. Empirical basis: measured across 4 real WC2022 GS matches (10503/3853/3855/
    10517) -> {1.11, 1.25, 1.63, 1.01}, range [1.01, 1.63], mean 1.25 sd 0.27. The band
    [0.5, 3.0] gives ~2x headroom below the min (catches a domain-collapse / all-zero
    geometry regression -> mean toward 0) and ~1.8x above the max (catches a
    mirror/interval over-count -> mean roughly doubles past 3.0), while every observed
    clean match passes comfortably. A domain-size floor guards the 'mean stays ~1 but
    the in-domain set collapsed' mode the mean alone would miss."""
    _actions, packed = gs_match
    made = packed["packing_made"].dropna().astype(float)
    assert len(made) > 300, f"packing in-domain set collapsed to {len(made)} rows"
    made_mean = float(made.mean())
    assert 0.5 < made_mean < 3.0, f"packing_made mean {made_mean:.3f} outside sane band [0.5, 3.0]"


def test_report_practitioner_anchors(gs_match, capsys):
    """The MSC practitioner anchors below are league-specific -> REPORTED, not gated
    (plan B11 / review minor 10): mean-bypassed-per-packing-action (~2), goal-threat
    pts/shot (~8), % goals with a packing action in the preceding possession (67.4%).
    The per-action packing_made mean IS now gated -- see
    test_packing_made_mean_in_sane_band. Paste the reported anchors into the PR body."""
    from silly_kicks.spadl.utils import add_possessions

    _actions, packed = gs_match
    made = packed["packing_made"]
    made_mean = made.dropna().astype(float).mean()
    positive = made.dropna().astype(float)
    positive = positive[positive > 0]
    bypassed_mean = positive.mean() if len(positive) else float("nan")

    shot_ids = [spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")]
    n_shots = int(np.isin(packed["type_id"].to_numpy(), shot_ids).sum())
    gt_per_shot = packed["packing_goal_threat"].dropna().astype(float).sum() / max(n_shots, 1)

    acts = add_possessions(packed)
    goals = acts[
        np.isin(acts["type_id"].to_numpy(), shot_ids)
        & (acts["result_id"].to_numpy() == spadlconfig.result_id["success"])
    ]
    with_packing = 0
    for _, g in goals.iterrows():
        prior = acts[
            (acts["possession_id"] == g["possession_id"])
            & (acts["game_id"] == g["game_id"])
            & (acts["action_id"] < g["action_id"])
        ]
        if (prior["packing_made"].fillna(0) >= 1).any():
            with_packing += 1
    goal_pct = with_packing / len(goals) if len(goals) else float("nan")

    d_mask = packed["type_id"].to_numpy() == spadlconfig.actiontype_id["dribble"]
    dribbles = int(d_mask.sum())
    packed_dribble_domain = int((d_mask & (packed["result_id"].to_numpy() == spadlconfig.result_id["success"])).sum())

    with capsys.disabled():
        print(f"\n[TF-49 GS e2e report, match {_MATCH_ID}]")
        print(f"  packing_made per-action mean (non-NA rows): {made_mean:.3f}")
        print(f"  mean bypassed per packing-positive action:  {bypassed_mean:.3f}  (MSC anchor ~2)")
        print(f"  goal-threat points per shot:                {gt_per_shot:.3f}  (anchor ~8)")
        print(f"  goals w/ packing in possession: {with_packing}/{len(goals)} = {goal_pct:.1%}  (anchor 67.4%)")
        print(f"  GS dribbles in packing domain: {packed_dribble_domain}/{dribbles} (ballCarryOutcome R->success)")
    assert np.isfinite(made_mean)
