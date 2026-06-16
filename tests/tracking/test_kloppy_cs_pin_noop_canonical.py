"""Task 3.2 / ADR-031: the CS-pin is a NO-OP on already-canonical data (never double-inverts).

This is why the fix is a coordinate-system pin, NOT a blanket ``y = 68 - y`` flip: a blanket flip
would invert an already-canonical provider. Both real kloppy providers (SkillCorner, Metrica) are
y-inverted natively (Gate A), so the no-op property is guarded with REAL data **canonicalized**: feed
``convert_to_frames`` a real SkillCorner dataset pre-transformed to the canonical coordinate system,
and assert the acting player still lands at the real action ``start_y`` (canonical), NOT ``68 - start_y``.
A blanket flip in the gateway would invert this canonical input and fail here -- which the inverted-
provider y-identity golden (``test_kloppy_y_identity_golden``) cannot catch on its own.
"""

import json
from pathlib import Path

import pytest

_FIX = Path(__file__).resolve().parent.parent / "datasets" / "tracking" / "yident" / "skillcorner"


def test_cs_pin_is_noop_on_already_canonical_input():
    skillcorner = pytest.importorskip("kloppy.skillcorner")
    from silly_kicks.spadl._kloppy_coordinates import socceraction_coordinate_system
    from silly_kicks.tracking import kloppy as tracking_kloppy

    ref = json.loads((_FIX / "action_ref.json").read_text(encoding="utf-8"))
    ds = skillcorner.load(
        meta_data=str(_FIX / "match.json"),
        raw_data=str(_FIX / "tracking_slice.jsonl"),
        include_empty_frames=False,
    )
    # Pre-canonicalize: apply the SPADL coordinate system so the dataset is ALREADY canonical.
    canonical_ds = ds.transform(to_coordinate_system=socceraction_coordinate_system(ds.metadata))

    # convert_to_frames re-pins the CS; on already-canonical input it must be a NO-OP in y.
    frames, _ = tracking_kloppy.convert_to_frames(canonical_ds, output_convention="absolute_frame")
    players = frames[~frames["is_ball"].astype(bool)].copy()
    players["player_id"] = players["player_id"].astype(str)
    pp = players[(players["period_id"] == ref["period_id"]) & (players["player_id"] == str(ref["player_id"]))]
    assert not pp.empty
    j = (pp["time_seconds"] - ref["time_seconds"]).abs().idxmin()
    frame_y = float(pp.loc[j, "y"])

    # Already-canonical in -> canonical out (== action start_y). A blanket flip would give 68 - start_y.
    assert abs(frame_y - ref["start_y"]) < 1.5, (
        f"CS-pin is NOT a no-op on canonical input: frame_y={frame_y:.1f} vs start_y={ref['start_y']:.1f} "
        f"(blanket-flip would be {68.0 - frame_y:.1f}) -- a blanket flip would invert a clean provider"
    )
