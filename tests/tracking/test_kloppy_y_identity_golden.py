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
