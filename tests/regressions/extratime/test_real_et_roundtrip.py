"""ET round-trip fixtures (spec 2026-05-30 §6 review H, plan Task 8).

- **GS real data** (delivered ``gs_et/frames.parquet``, match 10517, period 3,
  ``meta.home_team_start_left_extratime=True``): the meta carries the *true* ET
  flag, so we assert **orientation correctness** (ET present + in SPADL bounds)
  with the flag, and the loud raise without it.
- **IDSSE (Sportec) / Metrica synthetic**: these have no ground-truth ET
  orientation, so they prove **e2e shape + no-crash + ET presence + bounds**, NOT
  orientation truth (review c) --- orientation reflection for all 5 converters is
  already asserted by ``test_et_guard_parity.py``. The deterministic synthesizer
  is the committed ``_builders.py`` (no separate parquet to drift; the builder IS
  the frozen artifact).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from regressions.extratime._builders import run_converter

GS = Path(__file__).resolve().parent / "gs_et"

# SPADL frame bounds (silly_kicks tracking + SPADL action coordinates).
_FIELD_LENGTH = 105.0
_FIELD_WIDTH = 68.0


# --- GS real-data ET round-trip (TF-23b: NATIVE GK + geometric ground truth) ------
#
# Fixture: GS WC2022 knockout match 10517, period 3 (first overtime), the RAW tracking-adapter
# input (x_centered + flags) carrying the NATIVE is_goalkeeper from the roster join — regenerated
# by scripts/regenerate_gs_et_native_gk.py. We do NOT trust meta.home_team_start_left_extratime
# (the constant GS-ET placeholder this feature exists to fix; for 10517 P3 it is geometrically
# WRONG — it leaves the home GK on the attacking half). Ground truth is GEOMETRIC: a defending GK
# sits deep in its OWN half, so in the canonical home-attacks-right (absolute) frame the home GK
# belongs at LOW x (<52.5). The backstop must achieve that regardless of the flag.


def _gs_inputs():
    frames = pd.read_parquet(GS / "frames.parquet")
    meta = pd.read_parquet(GS / "meta.parquet").iloc[0]
    return frames, meta, int(meta["home_team_id"])


def _home_gk_median_x(out, home_team_id):
    hg = out[(out["team_id"] == home_team_id) & (out["is_goalkeeper"]) & (~out["is_ball"])]
    assert len(hg) > 0, "no native home GK rows in converted output"
    return float(hg["x"].median())


def test_gs_real_et_native_gk_geometric_self_correction():
    """The geometric backstop self-corrects the (unreliable) placeholder ET flag: the native home
    GK lands on its defended LOW-x half under BOTH the placeholder flag and its negation, and the
    two converge — orientation is data-driven, not flag-driven."""
    from silly_kicks.tracking.gradientsports import convert_to_frames

    frames, meta, home_id = _gs_inputs()
    placeholder = bool(meta["home_team_start_left_extratime"])
    out_flag, _ = convert_to_frames(
        frames,
        home_team_id=home_id,
        home_team_start_left=bool(meta["home_start_left"]),
        home_team_start_left_extratime=placeholder,
        output_convention="absolute_frame",
    )
    out_neg, _ = convert_to_frames(
        frames,
        home_team_id=home_id,
        home_team_start_left=bool(meta["home_start_left"]),
        home_team_start_left_extratime=not placeholder,  # deliberately wrong
        output_convention="absolute_frame",
    )
    gk_flag = _home_gk_median_x(out_flag, home_id)
    gk_neg = _home_gk_median_x(out_neg, home_id)
    # Both conversions place the home GK on the defended (low-x) half (geometric truth).
    assert gk_flag < 52.5, f"home GK not on low-x half under placeholder flag: {gk_flag}"
    assert gk_neg < 52.5, f"home GK not self-corrected under negated flag: {gk_neg}"
    # And they converge: the flag no longer determines orientation (the net does).
    assert abs(gk_flag - gk_neg) < 0.01, f"flag still drives orientation: {gk_flag} vs {gk_neg}"
    # ET coords in SPADL bounds under both.
    for out in (out_flag, out_neg):
        et = out[out["period_id"] == 3]
        assert et["x"].between(0, _FIELD_LENGTH).all()
        assert et["y"].between(0, _FIELD_WIDTH).all()


def test_gs_real_et_raises_without_flag():
    from silly_kicks.tracking.gradientsports import convert_to_frames

    frames, meta, home_id = _gs_inputs()
    with pytest.raises(ValueError, match="ET periods"):
        convert_to_frames(
            frames,
            home_team_id=home_id,
            home_team_start_left=bool(meta["home_start_left"]),
            home_team_start_left_extratime=None,
            output_convention="absolute_frame",
        )


# --- IDSSE (Sportec) / Metrica synthetic ET round-trip ---------------------

_SYNTH_CASES = ("sportec_tracking", "sportec_events", "metrica_events")
_COORD = {
    "sportec_tracking": ("x", "y"),
    "sportec_events": ("start_x", "start_y"),
    "metrica_events": ("start_x", "start_y"),
}


@pytest.mark.parametrize("case", _SYNTH_CASES)
def test_synthetic_et_roundtrip_in_bounds_with_flag(case):
    out = run_converter(case, et=True, flag=True)
    et = out[out["period_id"].isin([3, 4])]
    assert len(et) > 0
    xcol, ycol = _COORD[case]
    assert et[xcol].dropna().between(0, _FIELD_LENGTH).all()
    assert et[ycol].dropna().between(0, _FIELD_WIDTH).all()


@pytest.mark.parametrize("case", _SYNTH_CASES)
def test_synthetic_et_raises_without_flag(case):
    with pytest.raises(ValueError, match="ET periods"):
        run_converter(case, et=True, flag=None)
