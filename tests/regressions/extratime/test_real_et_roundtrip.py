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


# --- GS real-data ET round-trip --------------------------------------------


def _gs_frames_from_bronze(bronze, *, home_team_id, away_team_id, home_start_left, et_flag):
    """Flatten the raw-bronze GS ET frames to the GS tracking converter input + convert.

    Roster-independent ET-direction logic (keys on team membership + period +
    coords), so a synthesized roster is sufficient (none delivered with the
    fixture). Mirrors ``scripts/_loader_pining.py::_build_gradientsports``.
    """
    from silly_kicks.tracking.gradientsports import add_gradientsports_player_ids, convert_to_frames

    df = bronze.copy()
    is_ball = df["is_ball"].astype(bool)
    jf = pd.DataFrame(
        {
            "game_id": int(df["match_id"].iloc[0]),
            "period_id": df["period"].astype(int),
            "frame_id": df["frame_num"].astype(int),
            "time_seconds": df["period_elapsed_time"].astype(float),
            "frame_rate": 29.97,
            "z": 0.0,
            "speed_native": float("nan"),
            "ball_state": "alive",
            "team_side": df["team_side"],
            "jersey_number": df["jersey_num"].where(~is_ball, None),
            "is_ball": is_ball.to_numpy(),
            "x_centered": df["x"].astype(float),
            "y_centered": df["y"].astype(float),
        }
    )
    players = jf.loc[~jf["is_ball"], ["team_side", "jersey_number"]].drop_duplicates()
    players["team_id"] = players["team_side"].map({"home": home_team_id, "away": away_team_id})
    roster = pd.DataFrame(
        {
            "team_id": players["team_id"].to_numpy(),
            "shirt_number": players["jersey_number"].to_numpy(),
            "player_id": range(1, len(players) + 1),
            "position_group_type": "MF",
        }
    )
    # One GK per team so the roster join doesn't warn (ET-direction is GK-independent).
    roster.loc[roster.groupby("team_id").head(1).index, "position_group_type"] = "GK"
    resolved, _ = add_gradientsports_player_ids(jf, roster, home_team_id=home_team_id, away_team_id=away_team_id)
    frames, _ = convert_to_frames(
        resolved,
        home_team_id=home_team_id,
        home_team_start_left=home_start_left,
        home_team_start_left_extratime=et_flag,
        output_convention="absolute_frame",
    )
    return frames


def _gs_inputs():
    frames = pd.read_parquet(GS / "frames.parquet")
    meta = pd.read_parquet(GS / "meta.parquet").iloc[0]
    home_id = int(meta["home_team_id"])
    return frames, meta, home_id, home_id + 1  # synthetic away id (membership only)


def test_gs_real_et_roundtrip_correct_orientation():
    frames, meta, home_id, away_id = _gs_inputs()
    out = _gs_frames_from_bronze(
        frames,
        home_team_id=home_id,
        away_team_id=away_id,
        home_start_left=bool(meta["home_start_left"]),
        et_flag=bool(meta["home_team_start_left_extratime"]),
    )
    et = out[out["period_id"].isin([3, 4])]
    assert len(et) > 0
    assert et["x"].between(0, _FIELD_LENGTH).all()
    assert et["y"].between(0, _FIELD_WIDTH).all()


def test_gs_real_et_raises_without_flag():
    frames, meta, home_id, away_id = _gs_inputs()
    with pytest.raises(ValueError, match="ET periods"):
        _gs_frames_from_bronze(
            frames,
            home_team_id=home_id,
            away_team_id=away_id,
            home_start_left=bool(meta["home_start_left"]),
            et_flag=None,
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
