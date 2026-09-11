"""Sample columns + provenance vocabulary for the TF-54b territorial-defense metric (spec §12)."""

from __future__ import annotations

#: Group keys for the per-(defender, match) samples table.
TD_SAMPLE_KEYS = ["game_id", "player_id"]

#: The metric + count + provenance columns emitted per (game_id, player_id).
TD_SAMPLE_COLUMNS = [
    "a_threat_suppressed",
    "a_frames_scored",
    "b_threat_suppressed",
    "b_frames_scored",
    "b_attribution_slippage",
    "td_source",
]

#: Closed provenance vocabulary of every EMITTED drop reason across both arms (ADR-042 -- never a
#: fabricated 0). ``scored`` = a real contribution. The per-sample ``td_source`` COLUMN carries the
#: Arm-A subset; the two Arm-B-only reasons (``missing_frame`` / ``non_finite_delta``) appear in
#: ``TerritorialDefenseReport.arm_b_drop_reasons``, not the column.
TD_SOURCE_VALUES = frozenset(
    {
        "scored",
        # Arm-A td_source reasons (also the per-sample column values):
        "no_actor",
        "no_defenders",
        "removal_undersupported",  # also an Arm-B census reason
        "unresolved_geometry",  # also an Arm-B census reason
        "fov_cropped_local",
        # Arm-B census-only reasons (report.arm_b_drop_reasons):
        "missing_frame",
        "non_finite_delta",
    }
)
