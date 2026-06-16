"""Cross-provider y-identity golden (PR-S94 Gate E, silly-kicks half; ADR-031).

Runs the LIVE tracking gateway (:func:`silly_kicks.tracking.kloppy.convert_to_frames`) on a committed
REAL provider slice and asserts the acting player's frame-y matches the real action ``start_y``,
restricted to OFF-CENTRE y (``|start_y-34|>8``) where the by-design action<->frame relation is
identity. RED before the CS-pin (tracking frames are y-inverted: ``frame_y == 68 - start_y``); GREEN
after. ``source_provider="synthetic"`` fixtures structurally cannot catch a self-consistent y-mirror,
so this uses minimal REAL captured data (see ``scripts/_capture_yident_sc.py``).

The action coordinates are committed REAL reference data (from the real SkillCorner events converter);
only ``convert_to_frames`` -- the code under test -- runs live here.
"""

import json
from pathlib import Path

import pytest

_FIX = Path(__file__).resolve().parent.parent / "datasets" / "tracking" / "yident"


def _load_skillcorner_frames(provider_dir: Path):
    skillcorner = pytest.importorskip("kloppy.skillcorner")
    from silly_kicks.tracking import kloppy as tracking_kloppy

    ds = skillcorner.load(
        meta_data=str(provider_dir / "match.json"),
        raw_data=str(provider_dir / "tracking_slice.jsonl"),
        include_empty_frames=False,
    )
    frames, _ = tracking_kloppy.convert_to_frames(ds, output_convention="absolute_frame")
    return frames


_LOADERS = {"skillcorner": _load_skillcorner_frames}


@pytest.mark.parametrize("provider", sorted(_LOADERS))
def test_acting_player_frame_y_matches_action_off_centre(provider):
    """Acting player's tracked frame-y == the action start_y (identity), NOT 68 - start_y."""
    provider_dir = _FIX / provider
    ref = json.loads((provider_dir / "action_ref.json").read_text(encoding="utf-8"))

    # Guard: the reference action must be off-centre, else identity vs y-flip are indistinguishable.
    assert abs(ref["start_y"] - 34.0) > 8.0, "reference action is not off-centre -- fixture invalid"

    frames = _LOADERS[provider](provider_dir)
    players = frames[~frames["is_ball"].astype(bool)].copy()
    players["player_id"] = players["player_id"].astype(str)
    pp = players[(players["period_id"] == ref["period_id"]) & (players["player_id"] == str(ref["player_id"]))]
    assert not pp.empty, f"{provider}: acting player not present in the tracking slice"

    dt = (pp["time_seconds"] - ref["time_seconds"]).abs()
    j = dt.idxmin()
    assert dt.loc[j] < 0.3, f"{provider}: no frame within 0.3s of the action ({dt.loc[j]:.2f}s)"
    frame_y = float(pp.loc[j, "y"])

    assert abs(frame_y - ref["start_y"]) < 1.5, (
        f"{provider}: acting-player frame_y={frame_y:.1f} != action start_y={ref['start_y']:.1f} "
        f"(68-flip would be {68.0 - frame_y:.1f}); tracking y is inverted vs the SPADL action y-axis"
    )


# ---------------------------------------------------------------------------
# IDSSE / Sportec: the native DFL parse+shape port -> native converters path
# (PR-S95 / ADR-031 T3). This is the production replacement for the retired
# kloppy `_kloppy_tracking_to_frames` dev-loader path. The off-centre actions in
# the committed slice happen to all be AWAY-team (early Bayern possession), so
# action-LTR and the absolute frame are a 180-degree reflection apart (ADR-028).
# We therefore use the library's per-action re-projection to bring the actor's
# frame position into action-LTR, after which acting-player y must equal the
# action start_y at IDENTITY -- uniformly for home and away. A y-inversion bug is
# orthogonal to the orientation flip, so it still fails this identity check.
# ---------------------------------------------------------------------------

_IDSSE_FIX = Path(__file__).resolve().parent.parent / "datasets" / "sportec" / "idsse_slice"


def _idsse_actions_frames():
    """Run the ACTUAL production DFL harness path on the committed slice.

    Uses ``scripts._loader_pining._build_idsse`` (the re-routed loader) so the test exercises the
    real port -> native-converter wiring, including the action team_id -> CLU remap that aligns the
    ADR-028 action<->frame join.
    """
    import scripts._loader_pining as loader

    paths = {
        "metadata": _IDSSE_FIX / "info.xml",
        "events": _IDSSE_FIX / "events.xml",
        "tracking": _IDSSE_FIX / "positions.xml",
    }
    actions, frames, _home = loader._build_idsse(paths, "DFL-MAT-J03WMX", None)
    return actions, frames


def test_idsse_acting_player_frame_y_matches_action_after_reprojection():
    """Acting player's frame position, re-projected into action-LTR (ADR-028), == the action
    start position at IDENTITY -- proving the native DFL path's tracking y agrees with its
    event y (the kloppy `_kloppy_tracking_to_frames` path inverted it; ADR-031)."""
    import pandas as pd

    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr

    actions, frames = _idsse_actions_frames()
    players = frames[~frames["is_ball"].astype(bool)].copy()
    players["player_id"] = players["player_id"].astype(str)

    cand = actions[(actions["start_y"] - 34.0).abs() > 8.0]
    cand = cand[cand["player_id"].notna()]
    assert not cand.empty, "no off-centre tracked actions in the IDSSE slice -- fixture invalid"

    rows = []
    for _, a in cand.iterrows():
        pp = players[(players["period_id"] == a["period_id"]) & (players["player_id"] == str(a["player_id"]))]
        if pp.empty:
            continue
        dt = (pp["time_seconds"] - a["time_seconds"]).abs()
        j = dt.idxmin()
        if dt.loc[j] >= 0.3:
            continue
        rows.append(
            {
                "game_id": a["game_id"],
                "period_id": a["period_id"],
                "team_id": a["team_id"],
                "frame_x": float(pp.loc[j, "x"]),
                "frame_y": float(pp.loc[j, "y"]),
                "start_x": float(a["start_x"]),
                "start_y": float(a["start_y"]),
            }
        )
    samp = pd.DataFrame(rows)
    assert not samp.empty, "no off-centre action linked to a frame within 0.3s -- fixture invalid"

    # Re-project the actor's frame position into each action's LTR frame (the library helper the
    # production context path uses), then identity must hold.
    flip = acting_team_attacks_rtl(samp, frames)
    samp_ltr = reproject_to_action_ltr(samp, flip, x_cols=["frame_x"], y_cols=["frame_y"])
    dy_ident = (samp_ltr["frame_y"] - samp_ltr["start_y"]).abs()
    # The y-inversion hypothesis: a frame y mirrored vs the action y-axis lands at (68 - start_y)
    # after re-projection. For these off-centre actions (|start_y-34| up to ~28m) an inversion is a
    # 20-56m error, vs the few-metres event-location-vs-tracked-actor offset on a clean action. So:
    # the best-matching off-centre action confirms identity within tracking noise, while its
    # y-inverted alternative is decisively far -- a robust discriminator that a tight blanket
    # tolerance (defeated by tackle event-location noise) is not.
    dy_inverted = (samp_ltr["frame_y"] - (68.0 - samp_ltr["start_y"])).abs()
    best = dy_ident.idxmin()
    assert dy_ident.loc[best] < 6.0 and dy_inverted.loc[best] > 15.0, (
        f"IDSSE: best off-centre action's acting-player frame_y disagrees with action start_y after "
        f"action-LTR re-projection (|dy_identity|={dy_ident.loc[best]:.2f}m, "
        f"|dy_inverted|={dy_inverted.loc[best]:.2f}m over {len(samp)} actions); the native DFL tracking "
        f"y is inverted vs the SPADL action y-axis"
    )
    # And typical (median) identity error stays well below the inversion scale.
    assert dy_ident.median() < dy_inverted.median(), (
        f"IDSSE: median identity error {dy_ident.median():.1f}m not below the y-inverted alternative "
        f"{dy_inverted.median():.1f}m -- tracking y looks inverted"
    )
