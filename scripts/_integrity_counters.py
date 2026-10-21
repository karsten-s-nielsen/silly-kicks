"""The ADR-126 integrity-gate inputs of ONE match, recorded as extract-stage counters (ADR-128 Task 20).

`scripts.validate_trained_artifact_integrity` bounds a plausible occurrence base rate by
``n_events * fps * horizon / n_rows``. Its inputs travel with the run: each xShot/xCross extract unit
records the label-domain event count (shots / crosses, selected by ``type_id`` exactly as the label
builder selects them, ADR-126) and the frame count and duration of the match's frames. They are counters
only (ADR-052 sidecars): they never enter a stage token or an artifact.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def integrity_counters(frames: pd.DataFrame, events: pd.DataFrame, type_names: Iterable[str]) -> dict[str, object]:
    """``{"n_events", "n_frames", "duration_s"}`` for one match: the ``events`` rows whose ``type_id`` is
    one of ``type_names``; the distinct ``(period_id, frame_id)`` of ``frames``; the summed per-period span
    of ``time_seconds`` (rounded to the microsecond, so the JSON sidecar is stable)."""
    from silly_kicks.spadl import config as spadlconfig

    ids = {spadlconfig.actiontype_id[t] for t in type_names}
    n_events = int(events["type_id"].isin(ids).sum()) if "type_id" in events.columns else 0
    keys = frames[["period_id", "frame_id"]].drop_duplicates()
    span = frames.groupby("period_id", observed=True)["time_seconds"].agg(lambda t: float(t.max() - t.min()))
    return {"n_events": n_events, "n_frames": len(keys), "duration_s": round(float(span.sum()), 6)}
