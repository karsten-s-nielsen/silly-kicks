"""Give a Gradient Sports test frame the possession-event clock (ADR-133, spec 2026-11-13 r8 §6.6 B).

The converter computes ``time_seconds = event_time - kickoff start_time`` per period, so every test input needs an
``event_time`` and one kickoff row per period. Older test builders write the intended period time as
``time_seconds`` (the old ``startGameClock`` input). :func:`with_event_clock` keeps that intent: period ``p``'s
kickoff sits at ``p * KICKOFF_BASE`` video seconds, and a row's ``event_time`` becomes kickoff + ``time_seconds``,
so the converter's output time equals the old input time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KICKOFF_BASE = 1000.0
KICKOFF_TYPE = {1: "FIRSTKICKOFF", 2: "SECONDKICKOFF", 3: "THIRDKICKOFF", 4: "FOURTHKICKOFF"}


def kickoff_time(period: int) -> float:
    return float(period) * KICKOFF_BASE


def with_event_clock(df: pd.DataFrame) -> pd.DataFrame:
    """A copy with ``event_time`` / ``start_time`` filled from ``time_seconds`` and a kickoff row per period 1-4.

    A row's own ``event_time`` wins; a NaN one becomes kickoff + ``time_seconds`` (still NaN where that is NaN).
    A NaN ``start_time`` becomes the row's ``event_time``. A period that already holds its kickoff row keeps it.
    """
    out = df.copy()
    period = out["period_id"].astype("int64")
    kick = period.map(kickoff_time).astype("float64")
    ts = out["time_seconds"].astype("float64") if "time_seconds" in out else pd.Series(np.nan, index=out.index)
    et = out["event_time"].astype("float64") if "event_time" in out else pd.Series(np.nan, index=out.index)
    out["event_time"] = et.where(et.notna(), kick + ts)
    st = out["start_time"].astype("float64") if "start_time" in out else pd.Series(np.nan, index=out.index)
    out["start_time"] = st.where(st.notna(), out["event_time"])

    have = set(zip(out["period_id"].astype("int64"), out["game_event_type"].fillna(""), strict=True))
    rows = []
    for p in sorted(set(period) & set(KICKOFF_TYPE)):
        if (p, KICKOFF_TYPE[p]) in have:
            continue
        row = dict.fromkeys(out.columns)
        row.update(
            {
                "game_id": out["game_id"].iloc[0],
                "event_id": -int(p),
                "period_id": int(p),
                "start_time": kickoff_time(p),
                "event_time": kickoff_time(p),
                "game_event_type": KICKOFF_TYPE[p],
            }
        )
        rows.append(row)
    if not rows:
        return out
    joined = pd.concat([pd.DataFrame(rows, columns=out.columns), out], ignore_index=True)
    for col in out.columns:
        try:
            joined[col] = joined[col].astype(out[col].dtype)
        except (TypeError, ValueError):
            pass
    return joined
