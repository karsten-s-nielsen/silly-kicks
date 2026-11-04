"""The committed IDSSE (DFL open data, CC-BY 4.0) event fixtures, converted through the Sportec converter."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from silly_kicks.spadl import sportec

IDSSE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "idsse"


def convert_idsse(match: str, *, drop_final_score: bool = False):
    """``sportec.convert_to_actions`` on one committed IDSSE match, oriented by its own meta.

    ``drop_final_score`` blanks ``whistle_final_result``, which forces the penalty reconciliation through its
    score-free steps (3 / 3m)."""
    meta = json.loads((IDSSE_DIR / f"{match}_meta.json").read_text(encoding="utf-8"))
    events = pd.read_parquet(IDSSE_DIR / f"{match}_events_native.parquet")
    if drop_final_score:
        events = events.copy()
        events["whistle_final_result"] = pd.NA
    return sportec.convert_to_actions(
        events,
        home_team_id="home",
        home_team_start_left=meta["home_team_start_left"],
        home_team_start_left_extratime=meta.get("home_team_start_left_extratime"),
    )
