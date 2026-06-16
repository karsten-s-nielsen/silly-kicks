"""IDSSE harness re-route invariance / N6 documentation (PR-S95 / ADR-031 T3).

Shows that the new DFL parse+shape-port path (``scripts._loader_pining._build_idsse`` ->
``providers.sportec`` -> native converters) UN-INVERTS the tracking y relative to the SPADL action
y-axis, versus the retired kloppy dev-loader path. The bug was an action<->frame y mismatch: the old
kloppy events gateway emitted canonical action y while the old loader-local ``_kloppy_tracking_to_frames``
emitted the kloppy-native (inverted) frame y. After the ADR-028 per-action LTR re-projection, the
acting player's frame y should equal the action ``start_y`` -- it does NOT on the old path (the bug),
and DOES on the new path (the fix). This is the N6 retrain-trigger documentation: IDSSE action-anchored
tracking-feature values change.

Old FRAMES come from the committed Phase-0 frozen golden (``idsse_oldpath_harness_golden.parquet``), so
this test never reconstructs the deleted ``_kloppy_tracking_to_frames``. The old ACTIONS use the kloppy
*events* gateway, which is NOT retired by this PR (it still serves other kloppy providers) -- hence the
``importorskip("kloppy.sportec")`` guard.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_FIX = Path(__file__).resolve().parents[1] / "datasets" / "sportec" / "idsse_slice"


def _best_offcentre_reproj_identity_dy(actions: pd.DataFrame, frames: pd.DataFrame) -> float | None:
    """Best (smallest) |frame_y - action start_y| over off-centre tracked actions, after the
    ADR-028 per-action LTR re-projection. Small => action and frame y agree (aligned)."""
    from silly_kicks.tracking._action_orientation import acting_team_attacks_rtl, reproject_to_action_ltr

    players = frames[~frames["is_ball"].astype(bool)].copy()
    players["player_id"] = players["player_id"].astype(str)
    cand = actions[(actions["start_y"] - 34.0).abs() > 8.0]
    cand = cand[cand["player_id"].notna()]
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
                "start_y": float(a["start_y"]),
            }
        )
    samp = pd.DataFrame(rows)
    if samp.empty:
        return None
    flip = acting_team_attacks_rtl(samp, frames)
    samp_ltr = reproject_to_action_ltr(samp, flip, x_cols=["frame_x"], y_cols=["frame_y"])
    return float((samp_ltr["frame_y"] - samp_ltr["start_y"]).abs().min())


def test_idsse_native_path_uninverts_y_vs_kloppy_old_path():
    sportec = pytest.importorskip("kloppy.sportec")
    import scripts._loader_pining as loader
    from silly_kicks.spadl import kloppy as spadl_kloppy

    # OLD path: kloppy events gateway (not retired) + the frozen kloppy-tracking frames golden.
    ev = sportec.load_event(event_data=str(_FIX / "events.xml"), meta_data=str(_FIX / "info.xml"))
    old_actions, _ = spadl_kloppy.convert_to_actions(ev)
    old_frames = pd.read_parquet(_FIX / "idsse_oldpath_harness_golden.parquet")
    old_dy = _best_offcentre_reproj_identity_dy(old_actions, old_frames)

    # NEW path: the re-routed loader (port -> native converters).
    paths = {"metadata": _FIX / "info.xml", "events": _FIX / "events.xml", "tracking": _FIX / "positions.xml"}
    new_actions, new_frames, _home = loader._build_idsse(paths, "DFL-MAT-J03WMX", None)
    new_dy = _best_offcentre_reproj_identity_dy(new_actions, new_frames)

    assert old_dy is not None, "old path produced no off-centre linked action -- fixture/measurement invalid"
    assert new_dy is not None, "new path produced no off-centre linked action -- fixture/measurement invalid"
    # The fix: new-path action y agrees with frame y essentially exactly (measured ~0.2 m) after
    # re-projection -- the native event + tracking converters share a y-axis.
    assert new_dy < 2.0, f"new path not y-aligned (best off-centre |dy|={new_dy:.1f}m)"
    # N6 -- the IDSSE harness feature values CHANGE: the old kloppy dev-loader path was meaningfully
    # MISaligned (measured ~11.8 m best-case; the loader-local kloppy-tracking frames disagreed with the
    # kloppy events gateway's action y-axis). This is IDSSE's partial misalignment, not the clean
    # full y-inversion that PR-S94 measured on SkillCorner -- but it is still a clear, retrain-triggering
    # change. The wide 2 m / 6 m gap is the robust discriminator.
    assert old_dy > 6.0, (
        f"old kloppy path expected MISaligned (the change being documented), but best off-centre "
        f"|dy|={old_dy:.1f}m -- if this is small the premise/fixture changed"
    )
    assert old_dy > new_dy + 4.0, f"new path ({new_dy:.1f}m) not meaningfully better-aligned than old ({old_dy:.1f}m)"
