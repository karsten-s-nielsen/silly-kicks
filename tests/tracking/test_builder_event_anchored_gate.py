"""Event-anchored action<->frame y-identity gate --- the primary Gate-C closer (SkillCorner).

Kloppy-independent: links SkillCorner SPADL actions to builder frames via silly-kicks'
link_actions_to_frames, reprojects to action-LTR (ADR-028), asserts the BALL position at
the action instant ~= the action start coordinate. The ball is identity-free (no
player-id bridge) and is the most on-target Gate-C probe (the historical bug was
builder-y vs event-y at the action location).

Committed fixtures are a real Databricks slice (post-join ``skillcorner_tracking`` +
``spadl_actions`` for one match; see meta.json ``source``). Metrica is NOT event-anchored
here --- its builder y is canonical-correct (validated dx=dy=0 vs the kloppy gateway in
test_builder_kloppy_parity_e2e), but the lakehouse metrica ``spadl_actions`` event-y did
not co-locate with the canonical tracking ball in a pilot (a lakehouse events data issue,
relayed separately). The kloppy-oracle parity is the proper metrica Gate-C check.
"""

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking import skillcorner as sk
from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr
from silly_kicks.tracking.utils import link_actions_to_frames

_FIX = Path(__file__).parent.parent / "datasets" / "tracking" / "sk_metrica_builder"
_TOL_M = 3.0  # action coord is the event tracker point; ~m tolerance
_CENTRE_BAND_M = 5.0  # exclude |y-34|<5: the |68-2y| error vanishes at centre

pytestmark = pytest.mark.skipif(
    not (_FIX / "meta.json").exists(),
    reason="TF-23 event-anchored fixtures not captured yet (DGX/Databricks step; see plan Task 6 Step 1)",
)


def _y_identity_residuals(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Per-action |ball_y - action_start_y| in the action-LTR frame, off-centre only.

    Uses the BALL position (identity-free) at the linked frame as the y-identity probe:
    for an on-ball action the ball is at the action start coord, and the ball needs no
    player-id bridge between the actions table and the builder's frames. The ball is
    reprojected into the acting team's LTR frame (ADR-028) before comparison.
    """
    pointers, _ = link_actions_to_frames(actions, frames, tolerance_seconds=0.2)
    linked = actions.merge(pointers[["action_id", "frame_id"]], on="action_id").dropna(subset=["frame_id"])
    linked["frame_id"] = linked["frame_id"].astype(int)  # pointers frame_id is float (NaN-able)
    ball = frames[frames["is_ball"].astype(bool)]
    merged = linked.merge(
        ball[["period_id", "frame_id", "x", "y"]],
        on=["period_id", "frame_id"],
        how="inner",
    ).reset_index(drop=True)
    # IMPORTANT: compute the flip mask on `merged` (post-merge index), NOT on `linked` ---
    # the merge resets the index, so a flip aligned to `linked` would misalign.
    flip = acting_team_attacks_rtl(merged, frames)
    merged = reproject_to_action_ltr(merged, flip, x_cols=["x"], y_cols=["y"])
    merged = merged[(merged["start_y"] - 34.0).abs() > _CENTRE_BAND_M]  # off-centre only
    merged = merged.dropna(subset=["x", "y"])  # ball may be untracked in some frames
    merged["dy"] = (merged["y"] - merged["start_y"]).abs()
    merged["dx"] = (merged["x"] - merged["start_x"]).abs()
    return merged


def test_skillcorner_event_anchored_y_identity_both_teams():
    """SkillCorner is the event-anchored Gate-C provider (real committed Databricks slice;
    actions + tracking share the native SkillCorner identity). Metrica is covered by the
    kloppy-oracle parity instead (test_builder_kloppy_parity_e2e) --- its builder y is
    canonical-correct (dx=dy=0 vs the gateway on open-data game 1), but the lakehouse
    metrica ``spadl_actions`` event-y did not co-locate with the canonical tracking ball in
    a pilot (a lakehouse events data issue, relayed separately; NOT the builder)."""
    import json

    bronze = pd.read_parquet(_FIX / "skillcorner_bronze.parquet")
    actions = pd.read_parquet(_FIX / "skillcorner_actions.parquet")
    home = str(json.loads((_FIX / "meta.json").read_text())["skillcorner_home_team_id"])
    frames, _ = sk.convert_to_frames(bronze, home_team_id=home)
    res = _y_identity_residuals(actions, frames)
    assert len(res) >= 4, "fixture must retain off-centre actions for both teams"
    # BOTH teams represented (catches a one-sided mirror that passes on a single team)
    assert res["team_id"].nunique() >= 2
    # PER-(team, period) medians --- NOT a global median: a single mis-oriented period
    # would slip under a global median, the exact subtle bug.
    grp = res.groupby(["team_id", "period_id"])[["dy", "dx"]].median()
    assert (grp["dy"] < _TOL_M).all(), f"skillcorner per-(team,period) y disagreement:\n{grp}"
    assert (grp["dx"] < _TOL_M).all(), f"skillcorner per-(team,period) x disagreement:\n{grp}"
