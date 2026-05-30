"""Shared RT/ET input builders + converter dispatch for the ET-direction tests.

Reuses each converter's existing minimal-input builder (DRY — no new shapes to drift)
and derives an ET variant by appending a period-3 copy. Used by:
  - capture_goldens.py (Task 0: RT-only goldens)
  - test_* (Tasks 4-8: guard raises, orientation parity, golden no-regress, round-trip)

Import via the `tests` pythonpath: `from regressions.extratime._builders import ...`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Reuse the converters' own minimal-input builders (tested shapes).
from spadl.test_gradientsports import _df_minimal_pass as _gs_events_rt  # type: ignore
from spadl.test_metrica import _df_minimal_pass as _metrica_events_rt  # type: ignore
from spadl.test_sportec import _df_minimal_pass as _sportec_events_rt  # type: ignore

_TRACK = Path(__file__).resolve().parents[2] / "datasets" / "tracking"

# (period column, home_team_id, away_team_id) per converter family.
_SPORTEC_TRACK_HOME = "DFL-CLU-0100"
_SPORTEC_EVENT_HOME = "DFL-CLU-A"
_METRICA_EVENT_HOME = "Home"
_GS_EVENT_HOME = 100


def _append_et(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """Append a period-3 (ET) copy of the first row(s), so the converter sees ET periods."""
    et = df.copy()
    et[period_col] = 3
    # nudge any frame/event id columns so they don't collide
    for c in ("frame_id", "frame_num", "event_id"):
        if c in et.columns and pd.api.types.is_numeric_dtype(et[c]):
            et[c] = et[c] + 10_000
    return pd.concat([df, et], ignore_index=True)


# ---- inputs ---------------------------------------------------------------


def sportec_tracking_input(*, et: bool) -> pd.DataFrame:
    df = pd.read_parquet(_TRACK / "sportec" / "tiny.parquet")
    return _append_et(df, "period_id") if et else df


def gs_tracking_input(*, et: bool) -> pd.DataFrame:
    df = pd.read_parquet(_TRACK / "gradientsports" / "tiny.parquet")
    return _append_et(df, "period_id") if et else df


def sportec_events_input(*, et: bool) -> pd.DataFrame:
    df = _sportec_events_rt()
    return _append_et(df, "period") if et else df


def metrica_events_input(*, et: bool) -> pd.DataFrame:
    df = _metrica_events_rt()
    return _append_et(df, "period") if et else df


def gs_events_input(*, et: bool) -> pd.DataFrame:
    df = _gs_events_rt()
    return _append_et(df, "period_id") if et else df


# ---- converter dispatch ---------------------------------------------------


def run_converter(case: str, *, et: bool, flag):
    """Run one converter on its RT/ET input with home_team_start_left_extratime=flag.

    `case` in {sportec_tracking, gs_tracking, sportec_events, gs_events, metrica_events}.
    Returns the converter's output DataFrame (frames or actions).
    """
    if case == "sportec_tracking":
        from silly_kicks.tracking.sportec import convert_to_frames

        out, _ = convert_to_frames(
            sportec_tracking_input(et=et),
            home_team_id=_SPORTEC_TRACK_HOME,
            home_team_start_left=True,
            home_team_start_left_extratime=flag,
            output_convention="absolute_frame",
        )
        return out
    if case == "gs_tracking":
        from silly_kicks.tracking.gradientsports import convert_to_frames

        df = gs_tracking_input(et=et)
        home = str(df.loc[~df["is_ball"], "team_id"].dropna().iloc[0])
        out, _ = convert_to_frames(
            df,
            home_team_id=home,
            home_team_start_left=True,
            home_team_start_left_extratime=flag,
            output_convention="absolute_frame",
        )
        return out
    if case == "sportec_events":
        from silly_kicks.spadl import sportec

        out, _ = sportec.convert_to_actions(
            sportec_events_input(et=et),
            home_team_id=_SPORTEC_EVENT_HOME,
            home_team_start_left=True,
            home_team_start_left_extratime=flag,
        )
        return out
    if case == "metrica_events":
        from silly_kicks.spadl import metrica

        out, _ = metrica.convert_to_actions(
            metrica_events_input(et=et),
            home_team_id=_METRICA_EVENT_HOME,
            home_team_start_left=True,
            home_team_start_left_extratime=flag,
        )
        return out
    if case == "gs_events":
        from silly_kicks.spadl import gradientsports

        out, _ = gradientsports.convert_to_actions(
            gs_events_input(et=et),
            home_team_id=_GS_EVENT_HOME,
            home_team_start_left=True,
            home_team_start_left_extratime=flag,
        )
        return out
    raise ValueError(f"unknown case {case!r}")


CASES = ("sportec_tracking", "gs_tracking", "sportec_events", "gs_events", "metrica_events")
