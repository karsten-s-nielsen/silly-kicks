"""Cross-provider ET-direction guard parity (spec 2026-05-30 §6, ADR-010).

All five per-period-absolute converters (Sportec + GS tracking; Sportec + GS +
Metrica events) must, on ET-without-flag:
  (a) raise the SAME exception type + standardized message shape (via the shared
      ``require_et_direction`` guard); and
  (b) actually orient ET when the flag IS provided --- the same ET row under
      ``flag=True`` vs ``flag=False`` reflects across the pitch
      (``x_left + x_right == 105``). This catches cross-provider drift where the
      guard fires identically but the post-guard flip math diverges.

Uses the shared RT/ET builders + ``run_converter`` dispatcher in ``_builders.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from regressions.extratime._builders import CASES, run_converter

# case -> the converter's `source` label embedded in the standardized message.
_SOURCE = {
    "sportec_tracking": "sportec convert_to_frames",
    "gs_tracking": "gradientsports convert_to_frames",
    "sportec_events": "sportec convert_to_actions",
    "gs_events": "gradientsports convert_to_actions",
    "metrica_events": "metrica convert_to_actions",
}

# tracking output uses `x`; SPADL events output uses `start_x`.
_COORD = {
    "sportec_tracking": "x",
    "gs_tracking": "x",
    "sportec_events": "start_x",
    "gs_events": "start_x",
    "metrica_events": "start_x",
}


@pytest.mark.parametrize("case", CASES)
def test_all_converters_raise_same_message_shape_on_et_without_flag(case):
    with pytest.raises(ValueError) as exc:
        run_converter(case, et=True, flag=None)
    msg = str(exc.value)
    src = _SOURCE[case]
    assert msg.startswith(f"{src}: data contains ET periods"), msg
    assert "home_team_start_left_extratime" in msg


_EVENTS = ("sportec_events", "gs_events", "metrica_events")
_TRACKING = ("sportec_tracking", "gs_tracking")


@pytest.mark.parametrize("case", _EVENTS)
def test_events_et_orientation_reflects_with_flag(case):
    out_left = run_converter(case, et=True, flag=True)
    out_right = run_converter(case, et=True, flag=False)
    coord = _COORD[case]

    # Align the ET subset by row position (same input + same order; only the flag
    # differs), then assert the matching rows reflect across the pitch.
    et_l = out_left[out_left["period_id"].isin([3, 4])].reset_index(drop=True)
    et_r = out_right[out_right["period_id"].isin([3, 4])].reset_index(drop=True)
    assert len(et_l) > 0 and len(et_l) == len(et_r)

    xl = et_l[coord].to_numpy(dtype="float64")
    xr = et_r[coord].to_numpy(dtype="float64")
    finite = np.isfinite(xl) & np.isfinite(xr)
    assert finite.any(), f"{case}: no finite ET coordinates to compare"
    assert np.allclose(xl[finite] + xr[finite], 105.0, atol=1e-6), (
        f"{case}: ET not reflected by flag: {xl[finite]} vs {xr[finite]}"
    )


@pytest.mark.parametrize("case", _TRACKING)
def test_tracking_et_self_corrects_regardless_of_flag(case):
    # TF-23b: the geometric backstop self-corrects a wrong ET flag, so flag=True and
    # flag=False converge to the SAME orientation (NOT reflected) -- net is tracking-only.
    out_left = run_converter(case, et=True, flag=True)
    out_right = run_converter(case, et=True, flag=False)
    et_l = out_left[out_left["period_id"].isin([3, 4])].reset_index(drop=True)
    et_r = out_right[out_right["period_id"].isin([3, 4])].reset_index(drop=True)
    assert len(et_l) > 0 and len(et_l) == len(et_r)

    xl = et_l["x"].to_numpy(dtype="float64")
    xr = et_r["x"].to_numpy(dtype="float64")
    finite = np.isfinite(xl) & np.isfinite(xr)
    assert finite.any(), f"{case}: no finite ET coordinates to compare"
    assert np.allclose(xl[finite], xr[finite], atol=1e-6), (
        f"{case}: ET not self-corrected: {xl[finite]} vs {xr[finite]}"
    )
